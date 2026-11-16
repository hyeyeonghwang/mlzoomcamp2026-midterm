
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
artifact = joblib.load("model/model.pkl")

model = artifact["model"]
classes = artifact["classes"]
features = artifact["features"]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    missing = [f for f in features if f not in data]
    if missing:
        return jsonify({"error": "Missing features", "fields": missing}), 400

    try:
        X = pd.DataFrame([data])[features]

        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]

        result = {
            "predicted_risk": classes[prediction],
            "probabilities": {
                label: round(float(prob), 4)
                for label, prob in zip(classes, probabilities)
            }
        }

        return jsonify(result)

    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
