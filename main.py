import os
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

example0 = {
    "dataset": {
        "class": QuickDrawDataset,
        "params": {
            "label_names": ["cat"],
            "download": True,
        },
    },
    "splits": [0.8, 0.1, 0.1],
    "batch_size": 256,
    "tokenizer": {
        "class": AbsolutePenPositionTokenizer, # DeltaPenPositionTokenizer
        "params": {
            "bins": 32,
        }
    },
    "model": {
        "class": SketchTransformer,
        "params": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 6,
            "max_len": 200,
        }
    },
    "training": {
        "num_epochs": 20,
        "learning_rate": 1e-4,
        "log_dir": "logs/sketch_transformer_example0",
        "resume": False,
    },
}

# TODO have a config system to select different configs
params = example0

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

batch_size = params["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialize model
model = params["model"]["class"]( **params["model"]["params"], vocab_size=len(tokenizer.vocab))

hparams = {
    "model_class": model.__class__.__name__,
    "num_layers": model.num_layers,
    "d_model": model.d_model,
    "nhead": model.nhead,
    "max_len": model.max_len,
    "tokenizer_class": tokenizer.__class__.__name__,
    "tokenizer_bins": tokenizer.bins,
    "learning_rate": params["training"]["learning_rate"],
}

start_epoch = 0

# Resume from checkpoint if available
checkpoint_path_prefix = f"{model.__class__.__name__}_{hparams.get('tokenizer_class')}-q{hparams.get('tokenizer_bins')}_checkpoint"
checkpoints = []

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