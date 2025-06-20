from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

#  load one model (adjust file name as needed)
MODEL_PATH = os.path.join("models", "xgb_beth_best.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]  # e.g., [1.2, 3.4, 5.6, ...]
    prediction = model.predict([features]).tolist()
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)