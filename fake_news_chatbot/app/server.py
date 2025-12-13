from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the project root directory (parent of app directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")

# Load model with error handling
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.json
        
        if not data:
            return jsonify({"error": "Request body is empty"}), 400

        if "text" not in data:
            return jsonify({"error": "text field missing"}), 400

        text = data["text"]
        
        if not isinstance(text, str):
            return jsonify({"error": "text must be a string"}), 400
        
        if not text.strip():
            return jsonify({"error": "text cannot be empty"}), 400

        # Make prediction
        prediction = model.predict([text])[0]
        confidence = model.predict_proba([text])[0]

        if prediction == 0:
            label = "FAKE"
            conf = confidence[0]
        else:
            label = "REAL"
            conf = confidence[1]

        return jsonify({
            "label": label,
            "confidence": round(conf * 100, 2)
        })
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fake News Detection API is running!"})

if __name__ == "__main__":
    app.run(debug=True)
