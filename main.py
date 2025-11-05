import os
import tomllib
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader, random_split
from dataset import QuickDrawDataset, SketchDataset
from tokenizers import AbsolutePenPositionTokenizer, DeltaPenPositionTokenizer
from models import SketchTransformer, SketchTransformerConditional
from runner import train_model, train_model_cond

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seed = 42
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed_all(seed)


CLASS_MAP = {
    "QuickDrawDataset": QuickDrawDataset,
    "SketchDataset": SketchDataset,
    "AbsolutePenPositionTokenizer": AbsolutePenPositionTokenizer,
    "DeltaPenPositionTokenizer": DeltaPenPositionTokenizer,
    "SketchTransformer": SketchTransformer,
    "SketchTransformerConditional": SketchTransformerConditional,
}

def load_config(path: str):
    """Load TOML config and resolve class references."""
    with open(path, "rb") as f:
        config = tomllib.load(f)

    # Resolve dataset class
    dataset_class = CLASS_MAP[config["dataset"]["class"]]
    dataset_params = {
        k: v for k, v in config["dataset"].items() if k != "class"
    }

    # Resolve tokenizer class
    tokenizer_class = CLASS_MAP[config["tokenizer"]["class"]]
    tokenizer_params = {
        k: v for k, v in config["tokenizer"].items() if k != "class"
    }

    # Resolve model class
    model_class = CLASS_MAP[config["model"]["class"]]
    model_params = {
        k: v for k, v in config["model"].items() if k != "class"
    }

    # Return a structured dictionary
    return {
        "dataset": {"class": dataset_class, "params": dataset_params},
        "tokenizer": {"class": tokenizer_class, "params": tokenizer_params},
        "model": {"class": model_class, "params": model_params},
        "splits": [
            config["splits"]["train"],
            config["splits"]["val"],
            config["splits"]["test"],
        ],
        "training": config["training"],
    }


def main(config_path: str):
    params = load_config(config_path)

    # Load dataset
    training_data = params["dataset"]["class"]( **params["dataset"]["params"])
    tokenizer = params["tokenizer"]["class"]( **params["tokenizer"]["params"])
    dataset = SketchDataset(training_data, tokenizer, max_len=params["model"]["params"]["max_len"])


    # Split dataset
    splits = params["splits"]
    train_size = int(splits[0] * len(dataset))
    val_size = int(splits[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    batch_size = params["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize model
    model = params["model"]["class"]( **params["model"]["params"], vocab_size=len(tokenizer.vocab))

    # Move model to the appropriate device
    model = model.to(device)

    hparams = {
        "model_class": model.__class__.__name__,
        "num_layers": model.num_layers,
        "d_model": model.d_model,
        "nhead": model.nhead,
        "max_len": model.max_len,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_bins": tokenizer.bins,
        "learning_rate": params["training"]["learning_rate"],
        #"use_jit": params["training"].get("use_jit", False),
    }

    start_epoch = 0

    # Resume from checkpoint if available
    checkpoint_path_prefix = f"{model.__class__.__name__}_{hparams.get('tokenizer_class')}-q{hparams.get('tokenizer_bins')}_checkpoint"
    checkpoints = []

    if os.path.exists(params["training"]["log_dir"]):
        for fname in os.listdir(params["training"]["log_dir"]):
            if fname.startswith(checkpoint_path_prefix):
                checkpoints.append(fname)

    if checkpoints and params["training"]["resume"]:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_checkpoint')[-1].split('.')[0]))
        checkpoint_path = os.path.join(params["training"]["log_dir"], latest_checkpoint)
        print(f"Loading checkpoint from {checkpoint_path}")
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        start_epoch = int(latest_checkpoint.split('_checkpoint')[-1].split('.')[0]) + 1
        print(f"Resumed from epoch {start_epoch}")


    # Train model (conditional or not)
    if params["model"]["class"] == SketchTransformer:
        train_model(
            model,
            train_loader,
            val_loader,
            vocab_size=len(tokenizer.vocab),
            epochs=params["training"]["num_epochs"],
            lr=params["training"]["learning_rate"],
            device=device,
            hparams=hparams,
            log_dir=params["training"]["log_dir"],
            tokenizer=tokenizer,
            start_epoch=start_epoch,
        )

    elif params["model"]["class"] == SketchTransformerConditional:
        train_model_cond(
            model,
            train_loader,
            val_loader,
            vocab_size=len(tokenizer.vocab),
            epochs=params["training"]["num_epochs"],
            lr=params["training"]["learning_rate"],
            device=device,
            hparams=hparams,
            log_dir=params["training"]["log_dir"],
            tokenizer=tokenizer,
            start_epoch=start_epoch,
        )




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/example_0.toml")
    args = parser.parse_args()

    if os.path.exists(args.config):
        main(args.config)
    else:
        print(f"Config file {args.config} not found.")
