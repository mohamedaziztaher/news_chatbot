# Fake News Detection Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-16.0-black.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An intelligent full-stack application that detects fake and real news articles using machine learning, featuring a modern web interface and OCR capabilities for image-based text extraction.

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API Documentation](#-api-documentation) • [Screenshots](#-screenshots)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Details](#-model-details)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project is a comprehensive fake news detection system that combines machine learning with a user-friendly web interface. It can analyze text directly or extract text from images using OCR technology, making it versatile for analyzing news articles from various sources including newspaper images.

### Key Highlights

- **99% Accuracy**: High-performing ML model using TF-IDF vectorization and Logistic Regression
- **Dual Interface**: Modern Next.js web app and command-line interface
- **OCR Integration**: Extract and analyze text from images using PaddleOCR
- **Real-time Analysis**: Fast inference with confidence scores
- **Multi-language Support**: OCR supports multiple languages
- **Responsive Design**: Beautiful, modern UI with dark/light theme support

## ✨ Features

### Backend Features
- 🤖 **Machine Learning Model**: TF-IDF + Logistic Regression classifier
- 🔌 **RESTful API**: Flask-based API with multiple endpoints
- 📸 **OCR Integration**: Extract text from images using PaddleOCR
- 🔍 **Text Preprocessing**: Advanced preprocessing for newspaper OCR
- ⚡ **Fast Inference**: Optimized prediction pipeline
- 🛡️ **Error Handling**: Robust error handling and validation

### Frontend Features
- 🎨 **Modern UI**: Beautiful, responsive design with tunnel effect
- 🌓 **Dark/Light Theme**: Toggle between themes
- 💬 **Chat History**: Multiple conversations with search functionality
- 📱 **Responsive**: Works on desktop and mobile devices
- 🖼️ **Image Upload**: Upload images for OCR-based analysis
- ⚡ **Real-time Updates**: Instant feedback and loading states

## 🏗️ Architecture

```
┌─────────────────┐
│   Next.js App   │  ← Frontend (React/TypeScript)
│   (Port 3000)   │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│  Flask API      │  ← Backend (Python)
│  (Port 5000)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Model       │  ← Trained Model (joblib)
│  (TF-IDF + LR)  │
└─────────────────┘
```

## 🚀 Installation

### Prerequisites

- **Python 3.8+** and pip
- **Node.js 18+** and npm
- **Git** (for cloning)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd NlpProject
```

### Step 2: Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd fake_news_chatbot
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**
   ```bash
   # On Linux/Mac:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Train the model:**
   ```bash
   python src/train_model.py
   ```
   
   This will create `models/fake_news_model.pkl` (may take a few minutes).

### Step 3: Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd ../web_chatbot
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Create environment file:**
   ```bash
   # Create .env.local file
   echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:5000" > .env.local
   ```

## ⚡ Quick Start

### Running the Complete System

1. **Start the Flask backend** (Terminal 1):
   ```bash
   cd fake_news_chatbot
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   python app/server.py
   ```
   The API will be available at `http://127.0.0.1:5000`

2. **Start the Next.js frontend** (Terminal 2):
   ```bash
   cd web_chatbot
   npm run dev
   ```
   The web app will be available at `http://localhost:3000`

3. **Open your browser** and navigate to `http://localhost:3000`

## 💻 Usage

### Web Interface

1. Click **"New Chat"** to start a conversation
2. Enter news text or upload an image
3. Click **"Send"** to analyze
4. View results with label (FAKE/REAL) and confidence score
5. Use the search bar to find previous conversations
6. Toggle dark/light theme using the icon in the top-right

### Command-Line Interface

```bash
cd fake_news_chatbot
source venv/bin/activate
python src/chatbot.py
```

Then enter news articles when prompted. Type `exit` to quit.

### Using the Prediction Module

```python
from src.predict import predict

result = predict("Your news article text here")
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']}%")
```

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

#### 2. Predict from Text
**POST** `/predict`

Analyze text for fake news.

**Request:**
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

#### 3. Predict from Image
**POST** `/predict-from-image`

Upload an image file for OCR extraction and analysis.

**Request:** `multipart/form-data`
- `image`: Image file (JPEG, PNG, etc.)
- `language_hints` (optional): Comma-separated language codes (e.g., "en,fr")

**Response:**
```json
{
  "label": "REAL",
  "confidence": 87.45,
  "extracted_text": "Full extracted text...",
  "preprocessed_text": "Cleaned text...",
  "ocr_engine_used": "paddleocr",
  "text_detections": 15,
  "is_reputable_source": true
}
```

#### 4. Predict from Base64 Image
**POST** `/predict-from-image-base64`

Analyze image from base64-encoded string.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "language_hints": ["en"]
}
```

#### 5. Preview OCR
**POST** `/preview-ocr`

Extract text from image without running fake news analysis.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,..."
}
```

#### 6. Supported Languages
**GET** `/supported-languages`

Get list of supported OCR languages.

**Response:**
```json
{
  "paddleocr": ["en", "fr", "de", ...],
  "note": "PaddleOCR supports many languages..."
}
```

### Example API Calls

**Using cURL:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists discover breakthrough in renewable energy."}'
```

**Using Python:**
```python
import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "Your news article here"}
response = requests.post(url, json=data)
print(response.json())
```

## 🤖 Model Details

### Architecture

The model uses a **Pipeline** approach:

1. **TF-IDF Vectorization**
   - Removes English stop words
   - Extracts up to 50,000 features
   - Converts text to numerical features

2. **Logistic Regression Classifier**
   - Binary classification (Fake = 0, Real = 1)
   - Maximum 300 iterations
   - Optimized for text classification

### Preprocessing

- URL removal
- Special character removal
- Lowercasing
- Whitespace normalization
- **Newspaper OCR preprocessing**: Removes metadata, dates, prices, and focuses on meaningful content

### Performance

- **Accuracy**: 99%
- **Precision**: 99% (Fake), 98% (Real)
- **Recall**: 99% (Fake), 98% (Real)
- **F1-Score**: 99% (Fake), 98% (Real)

### Dataset

Trained on a dataset containing:
- Fake news articles (`data/Fake.csv`)
- Real news articles (`data/True.csv`)

Data is split into 80% training and 20% test sets.

## 🛠️ Technologies Used

### Backend
- **Python 3.8+**: Core language
- **Flask**: Web framework
- **scikit-learn**: Machine learning
- **pandas**: Data manipulation
- **PaddleOCR**: OCR engine
- **Pillow**: Image processing
- **joblib**: Model serialization
- **numpy**: Numerical computations

### Frontend
- **Next.js 16**: React framework
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling
- **React Hooks**: State management

## 📁 Project Structure

```
NlpProject/
│
├── fake_news_chatbot/          # Backend (Python/Flask)
│   ├── app/
│   │   └── server.py           # Flask API server
│   ├── src/
│   │   ├── train_model.py      # Model training script
│   │   ├── predict.py           # Prediction module
│   │   ├── preprocess.py       # Text preprocessing
│   │   ├── chatbot.py          # CLI chatbot
│   │   ├── ocr_service.py      # OCR functionality
│   │   └── image_utils.py      # Image utilities
│   ├── data/
│   │   ├── Fake.csv            # Fake news dataset
│   │   └── True.csv            # Real news dataset
│   ├── models/
│   │   └── fake_news_model.pkl # Trained model (generated)
│   ├── requirements.txt        # Python dependencies
│   └── README.md               # Backend documentation
│
├── web_chatbot/                # Frontend (Next.js)
│   ├── app/
│   │   ├── components/         # React components
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── ChatMessage.tsx
│   │   │   ├── ChatSidebar.tsx
│   │   │   ├── ImageUpload.tsx
│   │   │   └── ThemeToggle.tsx
│   │   ├── page.tsx            # Main page
│   │   └── layout.tsx           # Root layout
│   ├── package.json            # Node.js dependencies
│   └── README.md               # Frontend documentation
│
├── screenshots/                # Project screenshots
├── README.md                   # This file
├── QUICK_START.md              # Quick start guide
└── .gitignore                  # Git ignore rules
```

## 📸 Screenshots

<div align="center">

### Web Interface
![Web Interface](screenshots/Screenshot%20From%202025-12-18%2009-27-29.png)

### Dark Theme
![Dark Theme](screenshots/Screenshot%20From%202025-12-18%2009-27-41.png)

### Image Analysis
![Image Analysis](screenshots/Screenshot%20From%202025-12-18%2009-27-48.png)

### Chat History
![Chat History](screenshots/Screenshot%20From%202025-12-18%2009-28-01.png)

### Results Display
![Results](screenshots/Screenshot%20From%202025-12-18%2009-30-45.png)

</div>

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript for frontend code
- Add comments for complex logic
- Update documentation as needed
- Test your changes before submitting


## 🙏 Acknowledgments

- Dataset used for training the model
- scikit-learn community for excellent ML tools
- Flask community for the web framework
- Next.js team for the amazing React framework
- PaddleOCR for OCR capabilities

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub.

---


