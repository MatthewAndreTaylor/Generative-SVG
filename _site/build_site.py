from pathlib import Path
import shutil
import tomllib

from flask import Flask, render_template


ROOT = Path(__file__).resolve().parent
STATIC_SRC = ROOT / "static"
TEMPLATE_DIR = ROOT / "templates"
CONFIG_PATH = ROOT / "config.toml"
DIST_DIR = ROOT / "dist"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))


def find_examples() -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    root = STATIC_SRC / "examples"

    for group_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True):
        images = sorted(image.relative_to(STATIC_SRC).as_posix() for image in group_dir.glob("*.svg"))
        if images:
            groups.append({"id": group_dir.name, "images": images})

    return groups


def load_site_config() -> dict:
    with CONFIG_PATH.open("rb") as handle:
        return tomllib.load(handle).get("params", {})


def build_site() -> None:
    site_config = load_site_config()
    examples = find_examples()

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True)
    shutil.copytree(STATIC_SRC, DIST_DIR / "static")

    with app.app_context():
        html = render_template(
            "index.html",
            **site_config,
            examples=examples,
        )

    (DIST_DIR / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    build_site()