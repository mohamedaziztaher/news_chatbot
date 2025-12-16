# Fake News Detection Chatbot

An intelligent NLP-based system that detects fake and real news articles using machine learning. This project implements a RESTful API and command-line interface for real-time fake news detection with high accuracy.

## 🎯 Features

- **High Accuracy**: Achieves 99% accuracy on test data
- **RESTful API**: Flask-based web API for easy integration
- **CLI Interface**: Interactive command-line chatbot
- **Machine Learning Pipeline**: TF-IDF vectorization + Logistic Regression
- **Real-time Prediction**: Fast inference with confidence scores
- **Error Handling**: Robust error handling and input validation

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Training](#model-training)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd fake_news_chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment**
   ```bash
   # On Linux/Mac:
   source .venv/bin/activate
   
   # On Windows:
   .venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
fake_news_chatbot/
│
├── app/
│   └── server.py              # Flask REST API server
│
├── src/
│   ├── train_model.py         # Model training script
│   ├── predict.py             # Prediction module
│   ├── preprocess.py          # Text preprocessing utilities
│   └── chatbot.py             # CLI chatbot interface
│
├── data/
│   ├── Fake.csv               # Fake news dataset
│   └── True.csv               # Real news dataset
│
├── models/
│   └── fake_news_model.pkl    # Trained model (generated after training)
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 💻 Usage

### 1. Train the Model

Before using the API or chatbot, you need to train the model:

```bash
python src/train_model.py
```

This will:
- Load and preprocess the training data
- Train a TF-IDF + Logistic Regression model
- Evaluate the model performance
- Save the trained model to `models/fake_news_model.pkl`

**Expected Output:**
```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      4688
           1       0.98      0.98      0.98      4292

    accuracy                           0.99      8980
   macro avg       0.99      0.99      0.99      8980
weighted avg       0.99      0.99      0.99      8980

Model trained and saved as models/fake_news_model.pkl
```

### 2. Start the Flask API Server

```bash
python app/server.py
```

The server will start on `http://127.0.0.1:5000`

### 3. Use the CLI Chatbot

```bash
python src/chatbot.py
```

Interactive mode where you can enter news articles and get predictions.

## 📡 API Documentation

### Base URL
```
http://127.0.0.1:5000
```

### Endpoints

#### 1. Health Check
**GET** `/`

Check if the API is running.

**Response:**
```json
{
  "message": "Fake News Detection API is running!"
}
```

#### 2. Predict News
**POST** `/predict`

Predict whether a news article is fake or real.

**Request Body:**
```json
{
  "text": "Your news article text here..."
}
```

**Response:**
```json
{
  "label": "FAKE",
  "confidence": 92.27
}
```

**Example using cURL:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists have discovered a new breakthrough in renewable energy technology."}'
```

**Example using Python:**
```python
import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "Your news article here"}
response = requests.post(url, json=data)
print(response.json())
```

**Error Responses:**

- `400 Bad Request`: Missing or invalid input
  ```json
  {
    "error": "text field missing"
  }
  ```

- `500 Internal Server Error`: Prediction failed
  ```json
  {
    "error": "Prediction failed: <error message>"
  }
  ```

## 🤖 Model Training

### Model Architecture

The model uses a **Pipeline** approach combining:

1. **TF-IDF Vectorization**
   - Removes English stop words
   - Extracts up to 50,000 features
   - Converts text to numerical features

2. **Logistic Regression Classifier**
   - Binary classification (Fake = 0, Real = 1)
   - Maximum 300 iterations

### Preprocessing

The training pipeline includes:
- URL removal
- Special character removal
- Lowercasing
- Whitespace normalization

**OCR Text Preprocessing:**
For text extracted from newspaper images, additional preprocessing is applied:
- Removes newspaper metadata (dates, weather info, prices, website URLs)
- Filters out section labels and newspaper names
- Extracts meaningful content (headlines, article snippets)
- Focuses on actual news content for better prediction accuracy

### Dataset

The model is trained on a dataset containing:
- Fake news articles (`data/Fake.csv`)
- Real news articles (`data/True.csv`)

The data is shuffled and split into:
- 80% training set
- 20% test set

### Performance

- **Accuracy**: 99%
- **Precision**: 99% (Fake), 98% (Real)
- **Recall**: 99% (Fake), 98% (Real)
- **F1-Score**: 99% (Fake), 98% (Real)

## 🛠️ Technologies Used

- **Python 3.8+**: Programming language
- **Flask**: Web framework for REST API
- **scikit-learn**: Machine learning library
  - `TfidfVectorizer`: Text vectorization
  - `LogisticRegression`: Classification model
  - `Pipeline`: Model pipeline
- **pandas**: Data manipulation and analysis
- **joblib**: Model serialization
- **numpy**: Numerical computations
- **PaddleOCR**: OCR engine for text extraction from images
- **Pillow**: Image processing

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:

```
flask>=2.3.0
flask-cors>=4.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
joblib>=1.3.0
numpy>=1.24.0
Pillow>=10.0.0
paddleocr[doc-parser]
paddlepaddle
```

## 🔧 Development

### Running Tests

You can test the prediction module directly:

```bash
python src/predict.py
```

### Using the Prediction Module

```python
from src.predict import predict

result = predict("Your news article text here")
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']}%")
```

### Text Preprocessing

The preprocessing module provides utility functions:

```python
from src.preprocess import clean_text

cleaned = clean_text("Your text with URLs and special chars!")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset used for training the model
- scikit-learn community for excellent ML tools
- Flask community for the web framework

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a development server. For production deployment, use a production WSGI server like Gunicorn or uWSGI.

