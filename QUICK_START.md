# Quick Start Guide

## Running the Complete System

### Step 1: Start the Flask Backend

Open a terminal and run:

```bash
cd /path/to/fake_news_chatbot
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python app/server.py
```

The Flask server will start on `http://127.0.0.1:5000`

### Step 2: Start the Next.js Frontend

Open a **new terminal** and run:

```bash
cd /path/to/fake_news_chatbot/web-chatbot
npm install  # Only needed the first time
npm run dev
```

The web interface will start on `http://localhost:3000`

### Step 3: Use the Application

1. Open your browser and go to `http://localhost:3000`
2. Click "New Chat" to start
3. Enter a news article or headline
4. Click "Send" to analyze
5. View the results (FAKE/REAL with confidence score)

## Environment Setup

### Flask Backend
- Virtual environment: `.venv`
- Dependencies: Installed via `requirements.txt`
- Port: `5000` (default)

### Next.js Frontend
- Create `.env.local` file in `web-chatbot/` directory:
  ```
  NEXT_PUBLIC_API_URL=http://127.0.0.1:5000
  ```
- Port: `3000` (default)

## Troubleshooting

### Port Already in Use
- Flask: Change port in `app/server.py` (last line)
- Next.js: Use `npm run dev -- -p 3001` to use port 3001

### CORS Errors
- Flask-CORS is already installed and configured
- If issues persist, check that Flask server is running

### API Connection Failed
- Verify Flask server is running
- Check `NEXT_PUBLIC_API_URL` in `.env.local`
- Ensure both servers are running simultaneously

## Features

✅ **Web Interface**: Modern, responsive design
✅ **Dark/Light Theme**: Toggle between themes
✅ **Chat History**: Search and manage multiple chats
✅ **Real-time Analysis**: Instant fake news detection
✅ **High Accuracy**: 99% model accuracy

Enjoy using the Fake News Detection Chatbot! 🚀

