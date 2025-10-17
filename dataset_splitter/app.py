import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
from dataset import QuickDrawDataset, TUBerlinDataset, SketchyDataset

# pip install Flask>=3.0.0 pandas>=2.0.0

app = Flask(__name__)

# Load sketch_features metadata
with open("sketch_features.json", "r") as f:
    sketch_meta = json.load(f)

# Prepare dataset
label_names = list(sketch_meta.keys())

dataset = QuickDrawDataset(label_names, download=True)
# dataset = SketchyDataset(label_names, download=True)

labels = [dataset.label_map[data_label] for data_label in dataset.labels]

print(f"Loaded {len(dataset)} sketches from {len(label_names)} categories.")

CSV_PATH = f"{dataset.__class__.__name__.lower()}_marked.csv"

# Load or initialize CSV
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.DataFrame(
        columns=[
            "label_name",
            "label_id",
            "index",
            "recognizable",
            "feature_complete",
        ]
    )
    df.to_csv(CSV_PATH, index=False)

# Determine last index used
last_index = df["index"].max() if not df.empty else -1
current_index = int(last_index) + 1


@app.route("/")
def index():
    return render_template("index.html", labels=labels)


@app.route("/current", methods=["GET"])
def current_sketch():
    label = request.args.get("label")

    if label not in sketch_meta:
        return jsonify(error=f"Unknown label: {label}"), 400

    global current_index
    svg = dataset[current_index]

    return jsonify(label=label, idx=current_index, svg=svg)


@app.route("/update", methods=["POST"])
def update():
    global df

    data = request.json
    label = data["label"]
    recognizable = int(data["recognizable"])
    feature_complete = int(data["feature_complete"])
    index = int(data["index"])
    label_id = dataset.labels[index]

    # Remove old record for this index + label if exists
    df = df[~((df["label_name"] == label) & (df["index"] == index))]

    # Append new record
    new_entry = {
        "label_name": label,
        "label_id": label_id,
        "index": index,
        "recognizable": recognizable,
        "feature_complete": feature_complete,
    }

    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

    return jsonify(success=True, label=label)


@app.route("/marked_sketches", methods=["GET"])
def marked_sketches():
    label = request.args.get("label")

    subset = df[df["label_name"] == label]
    marked_sketches = []

    for _, row in subset.iterrows():
        marked_sketches.append(
            {
                "index": int(row["index"]),
                "svg": dataset[int(row["index"])],
                "recognizable": int(row["recognizable"]),
                "feature_complete": int(row["feature_complete"]),
            }
        )

    return jsonify(marked_sketches)


@app.route("/set_index", methods=["POST"])
def set_index():
    data = request.json
    index = data["index"]

    if index < 0 or index >= len(dataset):
        return jsonify(error="Invalid index"), 400

    global current_index
    current_index = index

    return jsonify(success=True, index=current_index)


@app.route("/get_index", methods=["GET"])
def get_index():
    global current_index
    return jsonify(index=current_index)


if __name__ == "__main__":
    app.run(debug=True)
