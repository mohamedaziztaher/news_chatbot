import joblib
import os
import sys

# Get the project root directory (parent of src directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")

# Global variable to cache the loaded model
_model = None

def load_model():
    """
    Load the trained fake news detection model.
    Caches the model in memory after first load.
    
    Returns:
        The loaded model pipeline
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    return _model

def predict(text):
    """
    Predict whether a given text is fake or real news.
    
    Args:
        text (str): The news text to analyze
    
    Returns:
        dict: Dictionary with 'label' (FAKE/REAL) and 'confidence' (percentage)
    
    Raises:
        ValueError: If text is empty or invalid
        Exception: If prediction fails
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    
    if not text.strip():
        raise ValueError("text cannot be empty")
    
    try:
        model = load_model()
        prediction = model.predict([text])[0]
        confidence = model.predict_proba([text])[0]
        
        if prediction == 0:
            label = "FAKE"
            conf = confidence[0]
        else:
            label = "REAL"
            conf = confidence[1]
        
        return {
            "label": label,
            "confidence": round(conf * 100, 2)
        }
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def analyze_news(text):
    """
    Alias for predict() function for backward compatibility.
    
    Args:
        text (str): The news text to analyze
    
    Returns:
        dict: Dictionary with 'result' (FAKE/REAL) and 'confidence' (percentage)
    """
    result = predict(text)
    return {
        "result": result["label"],
        "confidence": result["confidence"]
    }

if __name__ == "__main__":
    # Example usage
    test_text = "This is a sample news article to test the prediction."
    try:
        result = predict(test_text)
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']}%")
    except Exception as e:
        print(f"Error: {e}")

