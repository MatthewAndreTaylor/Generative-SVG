import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, render_template, request, jsonify
import json
from dataset import QuickDrawDataset

# pip install Flask>=3.0.0

app = Flask(__name__)

# Load sketch_features metadata
with open("sketch_features.json", "r") as f:
    sketch_meta = json.load(f)
    
    
with open("sketches.json", "r") as f:
    sketches = json.load(f)

label_names = list(sketch_meta.keys())
dataset = QuickDrawDataset(label_names, download=True)
labels = [dataset.label_map[data_label] for data_label in dataset.labels]


last_item = list(sketches.values())[-1]
last_index = max([int(s) for s in list(last_item.keys())], default=0)

current_index = last_index + 1
    

print(f"Loaded {len(dataset)} sketches from {len(label_names)} categories.")


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

    return jsonify(
        label=label,
        idx=current_index,
        svg=svg
    )

@app.route("/update", methods=["POST"])
def update():
    data = request.json
    index = data["index"]
    label = data["label"]
    recognizable = data["recognizable"]
    missing = data["feature_complete"]

    if label not in sketches:
        sketches[label] = {}

    sketches[label][index] = {
        "recognizable": recognizable,
        "feature_complete": missing
    }

    # Save back to JSON file
    with open("sketches.json", "w") as f:
        json.dump(sketches, f, indent=4)

    return jsonify(success=True, label=label)



@app.route("/marked_sketches", methods=["GET"])
def marked_sketches():

    label = request.args.get("label")

    if label not in sketches:
        return jsonify(error=f"Unknown label: {label}"), 400

    marked = sketches[label]
    marked_sketches = []

    for key, value in marked.items():
        marked_sketches.append({
            "index": int(key),
            "svg": dataset[int(key)],
            "recognizable": value["recognizable"],
            "feature_complete": value["feature_complete"]
        })

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