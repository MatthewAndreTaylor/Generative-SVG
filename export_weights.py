from pathlib import Path

import torch
import torch.onnx


CHECKPOINTS = {
    "small": Path("model_checkpoint_small.pt"),
    "medium": Path("model_checkpoint_medium.pt"),
}

OUTPUT_DIR = Path("_site/static/models")


def export_checkpoint(tag: str, checkpoint_path: Path) -> Path:
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.eval()

    dummy_tokens = torch.zeros(1, model.max_len, dtype=torch.long)
    dummy_label = torch.zeros(1, dtype=torch.long)
    output_path = OUTPUT_DIR / f"{tag}.onnx"

    torch.onnx.export(
        model,
        (dummy_tokens, dummy_label),
        output_path.as_posix(),
        input_names=["tokens", "class_labels"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )

    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for tag, checkpoint_path in CHECKPOINTS.items():
        output_path = export_checkpoint(tag, checkpoint_path)
        print(f"Exported {tag} -> {output_path}")


if __name__ == "__main__":
    main()