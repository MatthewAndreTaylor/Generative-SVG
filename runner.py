import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import top_k_filtering, top_p_filtering


def sample_sequence_feat(
    model,
    start_tokens,
    max_len=200,
    temperature=1.0,
    top_k=20,
    top_p=0.7,
    greedy=False,
    eos_id=None,
    device="cuda",
):
    model.eval()
    tokens = list(start_tokens)
    tokens_tensor = torch.tensor([tokens], device=device, dtype=torch.long)

    for _ in range(max_len - len(tokens)):
        with torch.no_grad():
            logits = model(tokens_tensor)
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
        if eos_id is not None and next_token == eos_id:
            break

        next_token_tensor = torch.tensor([[next_token]], device=device)
        tokens_tensor = torch.cat([tokens_tensor, next_token_tensor], dim=1)

    return tokens


def train_model(
    model,
    train_loader,
    val_loader,
    vocab_size,
    log_dir,
    epochs=15,
    lr=1e-4,
    device="cuda",
    hparams={},
    tokenizer=None,
    num_samples=5,
    start_epoch=0,
):
    """Training loop with validation and TensorBoard logging."""
    assert (
        tokenizer is not None
    ), "Tokenizer must be provided for logging target distributions."

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    pad_idx = tokenizer.vocab["PAD"]

    use_padding_mask = hparams.get("use_padding_mask", False)
    if use_padding_mask:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_hparams(hparams, {})

    input_ids, _, _ = next(iter(train_loader))
    writer.add_graph(model, input_ids.to(device))

    # Initial evaluation to log target token distribution (validation set)
    all_targets = []
    for _, target_ids, _ in tqdm(val_loader, desc="Initial Eval"):
        mask = target_ids != pad_idx
        all_targets.append(target_ids[mask].detach().cpu())

    all_targets = torch.cat(all_targets)
    writer.add_histogram("Targets/ValTokens", all_targets, 0)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for input_ids, target_ids, _ in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"
        ):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            src_key_padding_mask = (
                None if not use_padding_mask else (input_ids == pad_idx)
            )
            logits = model(input_ids, src_key_padding_mask=src_key_padding_mask)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_token_accuracy = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for input_ids, target_ids, _ in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"
            ):
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                src_key_padding_mask = (
                    None if not use_padding_mask else (input_ids == pad_idx)
                )

                logits = model(input_ids, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                val_loss += loss.item()

                # Calculate accuracy of next token predictions
                preds = logits.argmax(dim=-1)
                mask = target_ids != pad_idx
                correct = (preds[mask] == target_ids[mask]).float().sum()
                total = mask.sum()
                acc = (correct / total).item() if total > 0 else 0.0
                val_token_accuracy += acc

                # Collect predictions and targets for histogram logging
                all_preds.append(preds[mask].detach().cpu())

        all_preds = torch.cat(all_preds)

        # Log histograms of predictions vs targets
        writer.add_histogram("Predictions/ValTokens", all_preds, epoch)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_token_accuracy / len(val_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/ValNextToken", avg_val_acc, epoch)
        writer.add_scalar(
            "Perplexity/Val", torch.exp(torch.tensor(avg_val_loss)).item(), epoch
        )
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

    # save checkpoint
    torch.save(
        model,
        f"{log_dir}/{model.__class__.__name__}_{hparams.get('tokenizer_class')}-q{hparams.get('tokenizer_bins')}_checkpoint{epoch}.pth",
    )

    with torch.no_grad():
        for i in range(num_samples):
            generated = sample_sequence_feat(
                model,
                start_tokens=[tokenizer.vocab["START"]],
                max_len=200,
                temperature=1.0,
                greedy=False,
                eos_id=tokenizer.vocab["END"],
                device=device,
            )
            decoded_sketch = tokenizer.decode(generated, stroke_width=0.3)
            writer.add_text(f"Generations/Val_{i}", decoded_sketch, epoch)
    writer.close()


def sample_sequence_feat_cond(
    model,
    start_tokens,
    class_label,
    max_len=200,
    temperature=0.8,
    top_k=20,
    top_p=0.7,
    greedy=False,
    eos_id=None,
    device="cuda",
):
    model.eval()
    tokens = list(start_tokens)
    tokens_tensor = torch.tensor([tokens], device=device, dtype=torch.long)
    class_label_tensor = torch.tensor([class_label], device=device, dtype=torch.long)

    for _ in range(max_len - len(tokens)):
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
        if eos_id is not None and next_token == eos_id:
            break

        # Append new token for next iteration
        next_token_tensor = torch.tensor([[next_token]], device=device)
        tokens_tensor = torch.cat([tokens_tensor, next_token_tensor], dim=1)

    return tokens


def train_model_cond(
    model,
    train_loader,
    val_loader,
    vocab_size,
    log_dir,
    epochs=15,
    lr=1e-4,
    device="cuda",
    hparams={},
    tokenizer=None,
    num_samples=5,
    start_epoch=0,
):
    """Training loop with validation and TensorBoard logging."""
    assert (
        tokenizer is not None
    ), "Tokenizer must be provided for logging target distributions."

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_hparams(hparams, {})
    
    pad_idx = tokenizer.vocab["PAD"]
    use_padding_mask = hparams.get("use_padding_mask", False)
    if use_padding_mask:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    input_ids, _, class_labels = next(iter(train_loader))
    writer.add_graph(model, (input_ids.to(device), class_labels.to(device)))

    # Initial evaluation to log target token distribution (validation set)
    all_targets = []
    for _, target_ids, _ in tqdm(val_loader, desc="Initial Eval"):
        mask = target_ids != tokenizer.vocab["PAD"]
        all_targets.append(target_ids[mask].detach().cpu())

    all_targets = torch.cat(all_targets)
    writer.add_histogram("Targets/ValTokens", all_targets, 0)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for input_ids, target_ids, class_labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"
        ):
            input_ids, target_ids, class_labels = (
                input_ids.to(device),
                target_ids.to(device),
                class_labels.to(device),
            )
            src_key_padding_mask = (
                None if not use_padding_mask else (input_ids == pad_idx)
            )
            
            logits = model(input_ids, class_labels, src_key_padding_mask=src_key_padding_mask)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_token_accuracy = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for input_ids, target_ids, class_labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"
            ):
                input_ids, target_ids, class_labels = (
                    input_ids.to(device),
                    target_ids.to(device),
                    class_labels.to(device),
                )
                src_key_padding_mask = (
                    None if not use_padding_mask else (input_ids == pad_idx)
                )

                logits = model(input_ids, class_labels, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                val_loss += loss.item()

                # Calculate accuracy of next token predictions
                preds = logits.argmax(dim=-1)
                mask = target_ids != tokenizer.vocab["PAD"]
                correct = (preds[mask] == target_ids[mask]).float().sum()
                total = mask.sum()
                acc = (correct / total).item() if total > 0 else 0.0
                val_token_accuracy += acc

                # Collect predictions and targets for histogram logging
                all_preds.append(preds[mask].detach().cpu())

        all_preds = torch.cat(all_preds)

        # Log histograms of predictions vs targets
        writer.add_histogram("Predictions/ValTokens", all_preds, epoch)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_token_accuracy / len(val_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/ValNextToken", avg_val_acc, epoch)
        writer.add_scalar(
            "Perplexity/Val", torch.exp(torch.tensor(avg_val_loss)).item(), epoch
        )
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

    # save checkpoint
    torch.save(
        model,
        f"{log_dir}/{model.__class__.__name__}_{hparams.get('tokenizer_class')}-q{hparams.get('tokenizer_bins')}_checkpoint{epoch}.pth",
    )

    with torch.no_grad():
        for i in range(num_samples):
            generated = sample_sequence_feat(
                model,
                start_tokens=[tokenizer.vocab["START"]],
                max_len=200,
                temperature=1.0,
                greedy=False,
                eos_id=tokenizer.vocab["END"],
                device=device,
            )
            decoded_sketch = tokenizer.decode(generated, stroke_width=0.3)
            writer.add_text(f"Generations/Val_{i}", decoded_sketch, epoch)

    writer.close()
