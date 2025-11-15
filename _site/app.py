# add parent directory to sys.path for imports
import sys
import os

# Ensure parent project root is on sys.path regardless of CWD
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from utils import top_k_filtering, top_p_filtering
import os
import tomllib as toml_loader

# Device & model loading
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

try:
    model = torch.load("model_checkpoint.pt", map_location=device, weights_only=False)
except FileNotFoundError:
    model = None  # Will handle in endpoint
    print("Model checkpoint not found, running without model.")


def load_site_config(path: str = "config.toml"):
    """Load Hugo-style config.toml and adapt to a dict for Jinja.

    Returns minimal dictionary if file or parser unavailable.
    """
    if toml_loader is None or not os.path.isfile(path):
        return {
            "title": "Generative SVG",
            "authors": [],
            "pdf_url": None,
            "code_url": None,
            "demo_classes": ["bird", "crab", "guitar"],
            "abstract": "(abstract unavailable)",
            "representation": "(representation text)",
            "model": "(model description)",
            "training": "(training description)",
        }
    with open(path, "rb") as f:
        raw = toml_loader.load(f)

    params = raw.get("params", {})
    authors = params.get("authors", [])
    return {
        "site_title": raw.get("title", "Generative SVG"),
        "title": raw.get("title", "Generative SVG"),  # page title reuse
        "authors": authors,
        "pdf_url": params.get("pdf_url"),
        "code_url": params.get("code_url"),
        "demo_classes": params.get("demo_classes", ["bird", "crab", "guitar"]),
        "abstract": params.get("abstract", ""),
        "representation": params.get("representation", ""),
        "model": params.get("model", ""),
        "training": params.get("training", ""),
    }


site_config = load_site_config()
app = Flask(__name__)


def sample(
    model,
    start_tokens,
    eos_id,
    temperature=0.8,
    top_k=20,
    top_p=0.7,
    class_label=0,
):
    """Autoregressively sample tokens until EOS or max length.

    Args:
        model: The loaded PyTorch model with `max_len` attribute.
        start_tokens: List[int] initial token sequence.
        eos_id: Integer token signaling end of sequence.
        temperature: Softmax temperature.
        top_k: Keep only top_k tokens.
        top_p: Nucleus sampling cumulative probability threshold.
        class_label: Optional class conditioning label.
    Returns:
        List[int]: Generated token sequence including EOS (if reached).
    """
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
            next_token = torch.multinomial(probs, num_samples=1).item()

        tokens.append(next_token)
        if next_token == eos_id:
            break

        next_token_tensor = torch.tensor([[next_token]], device=device)
        tokens_tensor = torch.cat([tokens_tensor, next_token_tensor], dim=1)

    return tokens


@app.route("/", methods=["GET"])
def index():
    """Render main page using Jinja template."""
    return render_template(
        "index.html",
        title=site_config.get("title"),
        site_title=site_config.get("site_title", site_config.get("title")),
        authors=site_config.get("authors", []),
        pdf_url=site_config.get("pdf_url"),
        code_url=site_config.get("code_url"),
        demo_classes=site_config.get("demo_classes", []),
        abstract=site_config.get("abstract", ""),
        representation=site_config.get("representation", ""),
        model=site_config.get("model", ""),
        training=site_config.get("training", ""),
    )


@app.route("/sample", methods=["POST"])
def sample_endpoint():
    if model is None:
        return jsonify({"error": "Model checkpoint not found."}), 500

    data = request.get_json(force=True) or {}

    start_tokens = data.get("start_tokens")
    eos_id = data.get("eos_id")
    class_label = data.get("class_label", 0)
    temperature = data.get("temperature", 0.8)
    top_k = data.get("top_k", 20)
    top_p = data.get("top_p", 0.7)

    # Basic validation
    if not isinstance(start_tokens, list) or not all(
        isinstance(x, int) for x in start_tokens
    ):
        return jsonify({"error": "start_tokens must be a list of integers"}), 400
    if not isinstance(eos_id, int):
        return jsonify({"error": "eos_id must be an integer"}), 400

    if not isinstance(class_label, int) or not (0 <= class_label < model.num_classes):
        return jsonify({"error": "class_label must be an integer in [0, num_classes-1]"}), 400

    try:
        tokens = sample(
            model,
            start_tokens=start_tokens,
            eos_id=eos_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            class_label=class_label,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"tokens": tokens})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
