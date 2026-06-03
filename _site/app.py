import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from flask import Flask, render_template
import tomllib

site_config = {}

with open("config.toml", "rb") as f:
    site_config = tomllib.load(f).get("params", {})


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
            groups.append(
                {
                    "id": name,
                    "images": images,
                }
            )
    return groups


@app.route("/", methods=["GET"])
def index():
    """Render main page."""
    return render_template(
        "index.html",
        **site_config,
        examples=find_examples(),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
