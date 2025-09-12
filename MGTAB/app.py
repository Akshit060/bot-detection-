from flask import Flask, request, jsonify
import torch
import os

# Import your model class from train_gnn.py
from train_gnn import MGTABModel, build_data

app = Flask(__name__)

# -----------------------------
# Load model + dataset once at startup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
DATA_DIR = BASE_DIR
# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
config = checkpoint["config"]

# Recreate model with saved config
model = MGTABModel(
    input_dim=config["input_dim"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["layers"],
    dropout=config["dropout"],
    stance_num_classes=config["stance_num_classes"],
    num_relations=config["num_relations"],
)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Load dataset (for now, inference by node_id)
data = build_data(DATA_DIR)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return "Bot Detection API is running!"

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        req = request.json
        node_id = int(req.get("node_id"))

        with torch.no_grad():
            bot_logits, stance_logits = model(data.x, data.edge_index, data.edge_type, data.edge_weight)
            bot_prob = torch.sigmoid(bot_logits[node_id]).item()
            stance_pred = torch.argmax(stance_logits[node_id]).item()

        return jsonify({
            "node_id": node_id,
            "bot_probability": round(bot_prob, 4),
            "is_bot": bool(bot_prob >= 0.5),
            "stance_class": int(stance_pred)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
