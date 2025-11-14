import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from tqdm import tqdm
from dataset import SketchDataset
from utils import top_k_filtering, top_p_filtering

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seed = 42
torch.manual_seed(seed)

if device == "cuda":
    torch.cuda.manual_seed_all(seed)


def add_hparams(writer, param_dict, metrics_dict):
    exp, ssi, sei = hparams(param_dict, metrics_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metrics_dict.items():
        writer.add_scalar(k, v)


class SketchTrainer:
    def __init__(self, model, dataset, tokenizer, training_config: dict):
        self.training_config = training_config
        self.model = model.to(device)
        self.tokenizer = tokenizer

        sketch_dataset = SketchDataset(dataset, tokenizer, max_len=model.max_len)
        splits = training_config.get("splits", [0.8, 0.1, 0.1])

        train_size = int(splits[0] * len(sketch_dataset))
        val_size = int(splits[1] * len(sketch_dataset))
        test_size = len(sketch_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            sketch_dataset, [train_size, val_size, test_size]
        )

        batch_size = training_config.get("batch_size", 32)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        lr = training_config.get("learning_rate", 1e-4)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.use_padding_mask = training_config.get("use_padding_mask", False)
        self.criterion = nn.CrossEntropyLoss()
        if self.use_padding_mask:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id
            )

        # Logging hyperparameter setup
        self.hparams = {
            "model": model.__class__.__name__,
            "n_layers": model.num_layers,
            "d_model": model.d_model,
            "nhead": model.nhead,
            "max_len": model.max_len,
            "tokenizer": tokenizer.__class__.__name__,
            "tokenizer_bins": tokenizer.bins,
            "lr": lr,
            "use_padding_mask": training_config.get("use_padding_mask", False),
        }

        self.log_dir = training_config.get("log_dir", None)
        assert self.log_dir is not None, "log_dir must be specified in training_config"

        # Checkpointing setup
        self.load_from_checkpoint(training_config.get("checkpoint_path", None))
        self.writer = SummaryWriter(log_dir=self.log_dir_entry)
        add_hparams(self.writer, self.hparams, {})

        input_ids, _, class_labels = next(iter(self.train_loader))
        example_input = (input_ids.to(device), class_labels.to(device))
        self.writer.add_graph(model, example_input)

        # Initial evaluation to log target token distribution (validation set)
        all_targets = []
        for _, target_ids, _ in tqdm(self.val_loader, desc="Initial Eval"):
            mask = target_ids != self.tokenizer.pad_token_id
            all_targets.append(target_ids[mask].detach().cpu())

        all_targets = torch.cat(all_targets)
        self.writer.add_histogram("Targets/ValTokens", all_targets, 0)

    def load_from_checkpoint(self, checkpoint_path):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting fresh training.")
            self.log_dir_entry = os.path.join(
                self.log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.start_epoch = 0
            return

        self.start_epoch = int(
            os.path.basename(checkpoint_path).split("_")[-1].split(".")[0]
        )
        model = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(model.state_dict())
        self.log_dir_entry = os.path.dirname(os.path.abspath(checkpoint_path))
        print(f"Resumed training from checkpoint: {checkpoint_path}")

    def train(self, num_epochs: int):
        """Training loop with validation and TensorBoard logging."""
        model = self.model
        train_loader = self.train_loader
        val_loader = self.val_loader
        use_padding_mask = self.use_padding_mask
        optim = self.optim
        criterion = self.criterion
        epoch = self.start_epoch

        if use_padding_mask:

            def forward_pass(input_ids, class_labels):
                mask = input_ids == self.tokenizer.pad_token_id
                return model(input_ids, class_labels, src_key_padding_mask=mask)

        else:

            def forward_pass(input_ids, class_labels):
                return model(input_ids, class_labels, src_key_padding_mask=None)

        for epoch in range(self.start_epoch, num_epochs):
            model.train()
            total_loss = 0

            for input_ids, target_ids, class_labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"
            ):
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                class_labels = class_labels.to(device)

                logits = forward_pass(input_ids, class_labels)
                loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))

                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            self.model.eval()
            val_loss = 0
            val_token_accuracy = 0
            all_preds = []

            with torch.no_grad():
                for input_ids, target_ids, _ in tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"
                ):
                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)
                    class_labels = class_labels.to(device)

                    logits = forward_pass(input_ids, class_labels)
                    loss = criterion(
                        logits.view(-1, model.vocab_size), target_ids.view(-1)
                    )
                    val_loss += loss.item()

                    # Calculate accuracy of next token predictions
                    preds = logits.argmax(dim=-1)
                    mask = target_ids != self.tokenizer.pad_token_id
                    correct = (preds[mask] == target_ids[mask]).float().sum()
                    total = mask.sum()
                    acc = (correct / total).item() if total > 0 else 0.0
                    val_token_accuracy += acc

                    # Collect predictions and targets for histogram logging
                    all_preds.append(preds[mask].detach().cpu())

            all_preds = torch.cat(all_preds)

            # Log histograms of predictions vs targets
            self.writer.add_histogram("Predictions/ValTokens", all_preds, epoch)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_token_accuracy / len(val_loader)
            self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            self.writer.add_scalar("Loss/Val", avg_val_loss, epoch)
            self.writer.add_scalar("Accuracy/ValNextToken", avg_val_acc, epoch)
            self.writer.add_scalar(
                "Perplexity/Val", torch.exp(torch.tensor(avg_val_loss)).item(), epoch
            )
            print(
                f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

        self.save(num_epochs)

    def train_mixed(self, num_epochs: int):
        """Training loop with mixed precision, validation, and TensorBoard logging."""
        model = self.model
        train_loader = self.train_loader
        val_loader = self.val_loader
        use_padding_mask = self.use_padding_mask
        optim = self.optim
        criterion = self.criterion
        epoch = self.start_epoch

        # Mixed precision scaler (https://docs.pytorch.org/docs/stable/amp.html)
        scaler = torch.amp.GradScaler(device)

        if use_padding_mask:

            def forward_pass(input_ids, class_labels):
                mask = input_ids == self.tokenizer.pad_token_id
                return model(input_ids, class_labels, src_key_padding_mask=mask)

        else:

            def forward_pass(input_ids, class_labels):
                return model(input_ids, class_labels, src_key_padding_mask=None)

        for epoch in range(self.start_epoch, num_epochs):
            model.train()
            total_loss = 0

            for input_ids, target_ids, class_labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"
            ):
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                class_labels = class_labels.to(device)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = forward_pass(input_ids, class_labels)
                    loss = criterion(
                        logits.view(-1, model.vocab_size), target_ids.view(-1)
                    )

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                total_loss += loss.detach()

            avg_train_loss = total_loss / len(train_loader)
            self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)

            self.model.eval()
            val_loss = 0
            val_token_accuracy = 0
            all_preds = []

            with torch.no_grad():
                for input_ids, target_ids, class_labels in tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"
                ):
                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)
                    class_labels = class_labels.to(device)

                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = forward_pass(input_ids, class_labels)
                        loss = criterion(
                            logits.view(-1, model.vocab_size), target_ids.view(-1)
                        )

                    val_loss += loss.detach()

                    # Calculate accuracy of next token predictions
                    preds = logits.argmax(dim=-1)
                    mask = target_ids != self.tokenizer.pad_token_id
                    correct = (preds[mask] == target_ids[mask]).float().sum()
                    total = mask.sum()
                    acc = (correct / total).detach() if total > 0 else 0.0
                    val_token_accuracy += acc

                    # Collect predictions and targets for histogram logging
                    all_preds.append(preds[mask].detach().cpu())

            all_preds = torch.cat(all_preds)

            # Log histograms of predictions vs targets
            self.writer.add_histogram("Predictions/ValTokens", all_preds, epoch)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_token_accuracy / len(val_loader)
            self.writer.add_scalar("Loss/Val", avg_val_loss, epoch)
            self.writer.add_scalar("Accuracy/ValNextToken", avg_val_acc, epoch)
            self.writer.add_scalar("Perplexity/Val", torch.exp(avg_val_loss), epoch)
            print(
                f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

        self.save(num_epochs)

    def save(self, epoch):
        """Save the trained model to disk."""
        torch.save(self.model, os.path.join(self.log_dir_entry, f"model_{epoch}.pt"))
        
        with open(os.path.join(self.log_dir_entry, "hparams.json"), "w") as f:
            json.dump(self.hparams, f, indent=4)


# Note: sampling could be batched for effiecently generating multiple samples at once
def sample(
    model,
    start_tokens,
    eos_id,
    temperature=0.8,
    top_k=20,
    top_p=0.7,
    greedy=False,
    class_label=0,
):
    """Autoregressive sampling from the model given a starting token sequence."""
    model.eval()
    tokens = list(start_tokens)
    tokens_tensor = torch.tensor([tokens], device=device, dtype=torch.long)
    class_label_tensor = torch.tensor([class_label], device=device, dtype=torch.long)

    for _ in range(model.max_len - len(tokens)):
        with torch.no_grad():
            logits = model(tokens_tensor, class_label_tensor)
            next_logits = logits[:, -1, :] / temperature

            # top-k / top-p filtering
            next_logits = top_k_filtering(next_logits, top_k)
            next_logits = top_p_filtering(next_logits, top_p)
            probs = F.softmax(next_logits, dim=-1)
            if greedy:
                next_token = torch.argmax(probs, dim=-1).item()
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()

        tokens.append(next_token)
        if next_token == eos_id:
            break

        next_token_tensor = torch.tensor([[next_token]], device=device)
        tokens_tensor = torch.cat([tokens_tensor, next_token_tensor], dim=1)

    return tokens
