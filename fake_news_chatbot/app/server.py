from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import sys
import io
from PIL import Image
import logging

# Add src directory to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from ocr_service import extract_text, get_supported_languages
from image_utils import base64_to_image, validate_image
from preprocess import preprocess_text
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the project root directory (parent of app directory)
MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum image size (10MB)
MAX_IMAGE_SIZE = int(os.environ.get('MAX_IMAGE_SIZE', 10 * 1024 * 1024))

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

@app.route("/supported-languages", methods=["GET"])
def supported_languages():
    """Get list of supported OCR languages."""
    try:
        languages = get_supported_languages()
        return jsonify({
            "paddleocr": languages,
            "note": "PaddleOCR supports many languages. This is a subset of commonly used languages. The actual language support depends on the installed PaddleOCR models."
        })
    except Exception as e:
        logger.error(f"Error getting supported languages: {str(e)}")
        return jsonify({"error": f"Failed to get supported languages: {str(e)}"}), 500


@app.route("/predict-from-image", methods=["POST"])
def predict_from_image():
    """Extract text from uploaded image and analyze for fake news."""
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Get optional language hints
        language_hints = None
        language_param = request.form.get('language_hints', '')
        if language_param:
            # Comma-separated list of language codes
            language_hints = [lang.strip() for lang in language_param.split(',') if lang.strip()]
        
        # Read and validate image
        try:
            image_bytes = file.read()
            
            # Check file size
            if len(image_bytes) > MAX_IMAGE_SIZE:
                return jsonify({"error": f"Image too large. Maximum size is {MAX_IMAGE_SIZE / (1024*1024):.1f}MB"}), 400
            
            image = Image.open(io.BytesIO(image_bytes))
            
            if not validate_image(image):
                return jsonify({"error": "Invalid image format"}), 400
            
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            return jsonify({"error": f"Failed to read image: {str(e)}"}), 400
        
        # Extract text using PaddleOCR
        try:
            extracted_text, structured_text, ocr_metadata = extract_text(image, language_hints=language_hints)
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return jsonify({"error": f"OCR extraction failed: {str(e)}"}), 500
        
        if not extracted_text.strip():
            return jsonify({"error": "No text could be extracted from the image"}), 400
        
        # Preprocess OCR-extracted text to focus on meaningful news content
        preprocessed_text = preprocess_text(extracted_text, is_newspaper_ocr=True)
        
        if not preprocessed_text.strip():
            return jsonify({"error": "No meaningful text could be extracted from the image after preprocessing"}), 400
        
        # Enhance text with structured information if available to provide more context
        enhanced_text = preprocessed_text
        if structured_text:
            # Extract key information from structured text to add context
            lines = structured_text.split('\n')
            for line in lines:
                if 'Newspaper:' in line and 'Not identified' not in line:
                    newspaper_match = re.search(r'Newspaper:\s*(.+?)\.', line)
                    if newspaper_match:
                        newspaper_name = newspaper_match.group(1).strip().lower()
                        if newspaper_name not in enhanced_text.lower():
                            enhanced_text = f"{newspaper_name} {enhanced_text}"
                elif 'Main Story:' in line or 'Headline:' in line:
                    headline_match = re.search(r'(?:Main Story|Headline):\s*"(.+?)"', line)
                    if headline_match:
                        headline = headline_match.group(1).strip().lower()
                        if headline not in enhanced_text.lower():
                            enhanced_text = f"{enhanced_text} {headline}"
        
        # List of reputable newspapers
        reputable_newspapers = [
            'washington times', 'washington post', 'new york times', 
            'wall street journal', 'los angeles times', 'chicago tribune',
            'usa today', 'boston globe', 'the guardian', 'bbc', 'reuters',
            'associated press', 'ap news', 'bloomberg', 'forbes'
        ]
        
        text_lower = enhanced_text.lower()
        is_reputable_source = any(newspaper in text_lower for newspaper in reputable_newspapers)
        
        # Make prediction using enhanced text
        try:
            prediction = model.predict([enhanced_text])[0]
            confidence = model.predict_proba([enhanced_text])[0]
            
            if is_reputable_source and prediction == 0 and confidence[0] < 0.90:
                logger.info(f"Reputable source detected. Prediction: FAKE with {confidence[0]*100:.2f}% confidence")
            
            if prediction == 0:
                label = "FAKE"
                conf = confidence[0]
            else:
                label = "REAL"
                conf = confidence[1]
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        
        response = {
            "label": label,
            "confidence": round(conf * 100, 2),
            "extracted_text": extracted_text,
            "preprocessed_text": preprocessed_text,
            "ocr_engine_used": ocr_metadata.get('engine', 'paddleocr'),
            "text_detections": ocr_metadata.get('text_detections', 0),
            "language_hints": language_hints,
            "is_reputable_source": is_reputable_source
        }
        
        if structured_text:
            response["structured_text"] = structured_text
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Unexpected error in predict_from_image: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/predict-from-image-base64", methods=["POST"])
def predict_from_image_base64():
    """Extract text from base64-encoded image and analyze for fake news."""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.json
        
        if not data:
            return jsonify({"error": "Request body is empty"}), 400
        
        if "image" not in data:
            return jsonify({"error": "image field missing (base64 encoded)"}), 400
        
        # Get optional language hints
        language_hints = None
        if 'language_hints' in data:
            if isinstance(data['language_hints'], list):
                language_hints = data['language_hints']
            elif isinstance(data['language_hints'], str):
                language_hints = [lang.strip() for lang in data['language_hints'].split(',') if lang.strip()]
        
        # Decode base64 image
        try:
            image = base64_to_image(data["image"])
            
            if not validate_image(image):
                return jsonify({"error": "Invalid image format"}), 400
            
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Extract text using PaddleOCR
        try:
            extracted_text, structured_text, ocr_metadata = extract_text(image, language_hints=language_hints)
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return jsonify({"error": f"OCR extraction failed: {str(e)}"}), 500
        
        if not extracted_text.strip():
            return jsonify({"error": "No text could be extracted from the image"}), 400
        
        # Preprocess OCR-extracted text to focus on meaningful news content
        preprocessed_text = preprocess_text(extracted_text, is_newspaper_ocr=True)
        
        if not preprocessed_text.strip():
            return jsonify({"error": "No meaningful text could be extracted from the image after preprocessing"}), 400
        
        # Enhance text with structured information if available to provide more context
        enhanced_text = preprocessed_text
        if structured_text:
            # Extract key information from structured text to add context
            # Look for newspaper name and main headline to add credibility context
            lines = structured_text.split('\n')
            for line in lines:
                if 'Newspaper:' in line and 'Not identified' not in line:
                    # Extract newspaper name
                    newspaper_match = re.search(r'Newspaper:\s*(.+?)\.', line)
                    if newspaper_match:
                        newspaper_name = newspaper_match.group(1).strip().lower()
                        # Add newspaper name at the beginning for context
                        if newspaper_name not in enhanced_text.lower():
                            enhanced_text = f"{newspaper_name} {enhanced_text}"
                elif 'Main Story:' in line or 'Headline:' in line:
                    # Extract main headline for additional context
                    headline_match = re.search(r'(?:Main Story|Headline):\s*"(.+?)"', line)
                    if headline_match:
                        headline = headline_match.group(1).strip().lower()
                        # Add headline context if not already present
                        if headline not in enhanced_text.lower():
                            enhanced_text = f"{enhanced_text} {headline}"
        
        # List of reputable newspapers that are generally credible
        reputable_newspapers = [
            'washington times', 'washington post', 'new york times', 
            'wall street journal', 'los angeles times', 'chicago tribune',
            'usa today', 'boston globe', 'the guardian', 'bbc', 'reuters',
            'associated press', 'ap news', 'bloomberg', 'forbes'
        ]
        
        # Check if the text mentions a reputable newspaper
        text_lower = enhanced_text.lower()
        is_reputable_source = any(newspaper in text_lower for newspaper in reputable_newspapers)
        
        # Make prediction using enhanced text
        try:
            prediction = model.predict([enhanced_text])[0]
            confidence = model.predict_proba([enhanced_text])[0]
            
            # For reputable sources, if prediction is FAKE with confidence < 85%, 
            # flip to REAL since reputable sources are generally credible
            # This helps with the issue where headlines from real newspapers are misclassified
            if is_reputable_source and prediction == 0 and confidence[0] < 0.85:
                logger.info(f"Reputable source detected: {[n for n in reputable_newspapers if n in text_lower]}. Overriding FAKE prediction ({confidence[0]*100:.2f}% confidence) to REAL")
                prediction = 1  # Flip to REAL
                confidence = [1 - confidence[0], confidence[0]]  # Swap confidence values
            
            if prediction == 0:
                label = "FAKE"
                conf = confidence[0]
            else:
                label = "REAL"
                conf = confidence[1]
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        
        # List of reputable newspapers
        reputable_newspapers = [
            'washington times', 'washington post', 'new york times', 
            'wall street journal', 'los angeles times', 'chicago tribune',
            'usa today', 'boston globe', 'the guardian', 'bbc', 'reuters',
            'associated press', 'ap news', 'bloomberg', 'forbes'
        ]
        
        text_lower = enhanced_text.lower()
        is_reputable_source = any(newspaper in text_lower for newspaper in reputable_newspapers)
        
        # Make prediction using enhanced text
        try:
            prediction = model.predict([enhanced_text])[0]
            confidence = model.predict_proba([enhanced_text])[0]
            
            # For reputable sources, if prediction is FAKE with confidence < 85%, flip to REAL
            if is_reputable_source and prediction == 0 and confidence[0] < 0.85:
                logger.info(f"Reputable source detected. Overriding FAKE prediction to REAL")
                prediction = 1
                confidence = [1 - confidence[0], confidence[0]]
            
            if prediction == 0:
                label = "FAKE"
                conf = confidence[0]
            else:
                label = "REAL"
                conf = confidence[1]
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        
        response = {
            "label": label,
            "confidence": round(conf * 100, 2),
            "extracted_text": extracted_text,
            "preprocessed_text": preprocessed_text,
            "ocr_engine_used": ocr_metadata.get('engine', 'paddleocr'),
            "text_detections": ocr_metadata.get('text_detections', 0),
            "language_hints": language_hints,
            "is_reputable_source": is_reputable_source
        }
        
        if structured_text:
            response["structured_text"] = structured_text
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Unexpected error in predict_from_image_base64: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/preview-ocr", methods=["POST"])
def preview_ocr():
    """Preview OCR extraction without running fake news analysis."""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.json
        
        if not data or "image" not in data:
            return jsonify({"error": "image field missing (base64 encoded)"}), 400
        
        # Get optional language hints
        language_hints = None
        if 'language_hints' in data:
            if isinstance(data['language_hints'], list):
                language_hints = data['language_hints']
            elif isinstance(data['language_hints'], str):
                language_hints = [lang.strip() for lang in data['language_hints'].split(',') if lang.strip()]
        
        try:
            image = base64_to_image(data["image"])
            if not validate_image(image):
                return jsonify({"error": "Invalid image format"}), 400
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        try:
            extracted_text, structured_text, ocr_metadata = extract_text(image, language_hints=language_hints)
        except Exception as e:
            return jsonify({"error": f"OCR extraction failed: {str(e)}"}), 500
        
        response = {
            "extracted_text": extracted_text,
            "ocr_engine_used": ocr_metadata.get('engine', 'paddleocr'),
            "text_detections": ocr_metadata.get('text_detections', 0),
            "language_hints": language_hints
        }
        
        # Add structured text if available
        if structured_text:
            response["structured_text"] = structured_text
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in preview_ocr: {str(e)}")
        return jsonify({"error": f"Preview failed: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fake News Detection API is running!"})

if __name__ == "__main__":
    app.run(debug=True)
