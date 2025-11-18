import os
import tomllib
from argparse import ArgumentParser
from dataset import QuickDrawDataset, SketchDataset
from sketch_tokenizers import AbsolutePenPositionTokenizer, DeltaPenPositionTokenizer
from models import SketchTransformer, SketchTransformerConditional
from runner import SketchTrainer

CLASS_MAP = {
    "QuickDrawDataset": QuickDrawDataset,
    "SketchDataset": SketchDataset,
    "AbsolutePenPositionTokenizer": AbsolutePenPositionTokenizer,
    "DeltaPenPositionTokenizer": DeltaPenPositionTokenizer,
    "SketchTransformer": SketchTransformer,
    "SketchTransformerConditional": SketchTransformerConditional,
}


def load_config(path: str) -> SketchTrainer:
    """Load TOML config and resolve class references."""
    with open(path, "rb") as f:
        config = tomllib.load(f)

    # Resolve dataset class
    dataset_class = CLASS_MAP[config["dataset"]["class"]]
    dataset_params = {k: v for k, v in config["dataset"].items() if k != "class"}

    # Resolve tokenizer class
    tokenizer_class = CLASS_MAP[config["tokenizer"]["class"]]
    tokenizer_params = {k: v for k, v in config["tokenizer"].items() if k != "class"}

    # Resolve model class
    model_class = CLASS_MAP[config["model"]["class"]]
    model_params = {k: v for k, v in config["model"].items() if k != "class"}

    tokenizer = tokenizer_class(**tokenizer_params)
    model_params["vocab_size"] = len(tokenizer.vocab)
    model_params["num_classes"] = len(dataset_params["label_names"])
    model = model_class(**model_params)

    return SketchTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset_class(**dataset_params),
        training_config=config["training"],
    )


def main(config_path: str):
    """Main entry point for training models."""
    trainer = load_config(config_path)

    trainer.train_mixed(num_epochs=trainer.training_config.get("num_epochs", 10))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/example_0.toml")
    args = parser.parse_args()

    if os.path.exists(args.config):
        main(args.config)
    else:
        print(f"Config file {args.config} not found.")
