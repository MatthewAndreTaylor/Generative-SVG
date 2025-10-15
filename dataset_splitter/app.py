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
labels = dataset.labels

current_indices = {label: 0 for label in label_names}

print(f"Loaded {len(dataset)} sketches from {len(label_names)} categories.")


@app.route("/")
def index():
    return render_template("index.html", labels=labels)

@app.route("/next", methods=["GET"])
def next_sketch():
    label = request.args.get("label")

    if label not in sketches:
        return jsonify(error=f"Unknown label: {label}"), 400

    idx = current_indices[label]
    svg = sketches[label][idx]

    # Move index forward for next request
    current_indices[label] = (idx + 1) % len(sketches[label])

    return jsonify(
        label=label,
        index=idx,
        svg=svg,
        total=len(sketches[label])
    )

@app.route("/update", methods=["POST"])
def update():
    data = request.json
    index = data["index"]
    label = data["label"]
    recognizable = data["recognizable"]
    missing = data["missing_features"]
    
    sketches[label][index] = {}
    sketches[label][index]["recognizable"] = recognizable
    sketches[label][index]["missing_features"] = missing

    # Save back to JSON file
    with open("sketches.json", "w") as f:
        json.dump(sketches, f, indent=4)

    return jsonify(success=True, label=label)

if __name__ == "__main__":
    app.run(debug=True)