from flask import Flask, render_template, request, jsonify
import json
from ..dataset import QuickDrawDataset

app = Flask(__name__)

# Load sketches metadata
with open("sketches.json", "r") as f:
    sketches = json.load(f)

labels = list(sketches.keys())
svg_data = QuickDrawDataset(labels, download=True)

@app.route("/")
def index():
    return render_template(
        "index.html",
        sketches=sketches,
        svg_data=svg_data,
        labels=labels,
    )

@app.route("/update", methods=["POST"])
def update():
    data = request.json
    label = data["label"]
    recognizable = data["recognizable"]
    missing = data["missing_features"]

    sketches[label]["recognizable"] = recognizable
    sketches[label]["missing_features"] = missing

    # Save back to JSON file
    with open("sketches.json", "w") as f:
        json.dump(sketches, f, indent=4)

    return jsonify(success=True, label=label)

if __name__ == "__main__":
    app.run(debug=True)