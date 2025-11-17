import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from utils import top_k_filtering, top_p_filtering
import tomllib

# Note the device on the hosting site may not have a GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

site_config = {}
loaded_models = {}

with open("config.toml", "rb") as f:
    site_config = tomllib.load(f).get("params", {})

for model_info in site_config.get("models", []):
    checkpoint_path = model_info.get("checkpoint")
    model_tag = model_info.get("tag")
    try:
        loaded_models[model_tag] = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        loaded_models[model_tag] = None
        print(f"Loading model checkpoint {checkpoint_path} failed: {e}. Running without model.")


app = Flask(__name__)

def find_examples(root: str = os.path.join("static", "examples")):
    """Scan static examples directory and group example sketch paths."""
    groups = []
    for name in sorted(os.listdir(root), reverse=True):
        group_path = os.path.join(root, name)
        if not os.path.isdir(group_path):
            continue
        images = []
        for fn in os.listdir(group_path):
            if fn.lower().endswith(".svg"):
                images.append(os.path.join("examples", name, fn).replace("\\", "/"))
        if images:
            groups.append({
                "id": name,
                "images": images,
            })
    return groups


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
    """Render main page."""
    return render_template(
        "index.html",
        **site_config,
        examples=find_examples(),
    )


@app.route("/sample", methods=["POST"])
def sample_endpoint():
    """API endpoint to generate token sequences."""
    data = request.get_json(force=True) or {}
    start_tokens = data.get("start_tokens")
    eos_id = data.get("eos_id")
    class_label = data.get("class_label", 0)
    temperature = data.get("temperature", 0.8)
    top_k = data.get("top_k", 20)
    top_p = data.get("top_p", 0.7)

    req_model = data.get("model")
    inference_model = loaded_models.get(req_model)

    if inference_model is None:
        return jsonify({"error": "No inference model is available on the server."}), 503

    if not isinstance(start_tokens, list) or not all(
        isinstance(x, int) for x in start_tokens
    ):
        return jsonify({"error": "start_tokens must be a list of integers"}), 400
    if not isinstance(eos_id, int):
        return jsonify({"error": "eos_id must be an integer"}), 400

    if not isinstance(class_label, int) or not (0 <= class_label < inference_model.num_classes):
        return jsonify({"error": "class_label must be an integer in [0, num_classes-1]"}), 400

    try:
        tokens = sample(
            inference_model,
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
