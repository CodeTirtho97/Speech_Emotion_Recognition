# Speech Emotion Recognition

<img width="1024" height="1024" alt="Speech_Emotion_Recognition_Poster" src="https://github.com/user-attachments/assets/0a6831bf-796a-4baf-9fea-9d01ffdec049" />

A full-stack Speech Emotion Recognition (SER) system with ML backend and modern web interface that classifies emotions from speech audio files using the Toronto Emotional Speech Set (TESS) dataset.

## 🚀 Live Demo

- **Frontend:** [PLACEHOLDER - Add your deployed frontend URL]
- **Backend API:** [PLACEHOLDER - Add your deployed backend URL]

## ✨ Features

- 🎯 **92.75% Accuracy** - Production-quality SVM model
- 🎨 **Modern Web UI** - Responsive design with drag & drop upload
- 🎵 **Audio Preview** - Listen to audio before analysis
- 📊 **Confidence Scores** - View prediction confidence for all emotions
- 🚀 **REST API** - Complete Flask API for integration
- 📱 **Mobile Friendly** - Works on all devices

## 📊 Model Performance

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| **SVM**               | **92.75%** | **0.93** | **0.93** | **0.93** |
| Logistic Regression   | 89.86%   | 0.90     | 0.90   | 0.89     |
| KNN                   | 86.23%   | 0.87     | 0.86   | 0.86     |
| Random Forest         | 85.14%   | 0.85     | 0.85   | 0.85     |

## 🎭 Supported Emotions

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😲 Pleasant Surprise
- 😢 Sad

## 🛠️ Tech Stack

**Backend:**
- Python 3.12
- Flask + Flask-CORS
- Scikit-learn (SVM, PCA)
- Librosa (Audio Processing)
- SMOTE (Data Balancing)
- NumPy, Pandas

**Frontend:**
- HTML5 / CSS3
- Vanilla JavaScript
- Fetch API
- Responsive Design

## 📂 Project Structure

```
Speech_Emotion_Recognition/
├── backend/
│   ├── pipeline.py          # ML pipeline
│   ├── train_model.py       # Training script
│   ├── test_pipeline.py     # Testing script
│   ├── app.py               # Flask API
│   └── models/              # Saved models
├── frontend/
│   ├── index.html           # Main UI
│   ├── style.css            # Styling
│   └── script.js            # Frontend logic
├── Toronto Emotional Speech Set/  # Dataset
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone [PLACEHOLDER - Add your repository URL]
cd speech-emotion-recognition
```

### 2. Install Dependencies

**Important:** This project requires **Python 3.10-3.13** (Python 3.14+ not yet supported by librosa).

**Option A - Using pip:**
```bash
pip install -r requirements.txt
```

**Option B - Using conda (recommended):**
```bash
conda install -c conda-forge librosa scikit-learn flask flask-cors imbalanced-learn
```

**If you encounter issues:** Use Anaconda Python 3.10-3.13 which has all packages pre-configured.

### 3. Train Model (First Time Only)
```bash
python train_model.py
```

### 4. Start Backend
```bash
python app.py
```
Backend runs at: `http://localhost:5000`

### 5. Start Frontend
```bash
cd frontend
python -m http.server 8000
```
Frontend runs at: `http://localhost:8000`

### 6. Use the Application
1. Open browser to `http://localhost:8000`
2. Upload an audio file (WAV, MP3, OGG, FLAC)
3. Click "Analyze Emotion"
4. View results with confidence scores

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "SVM"
}
```

### Get Emotions List
```bash
GET /emotions
```

### Predict Emotion
```bash
POST /predict
Content-Type: multipart/form-data

file: <audio_file>
```
**Response:**
```json
{
  "success": true,
  "predicted_emotion": "Happy",
  "confidence_scores": {
    "Happy": 0.85,
    "Neutral": 0.10,
    ...
  },
  "model_used": "SVM"
}
```

## 📊 Dataset

- **Name:** Toronto Emotional Speech Set (TESS)
- **Size:** ~2,800 audio files
- **Format:** WAV (24,000 Hz)
- **Speakers:** 2 female speakers (ages 26 and 64)
- **Link:** [TESS Dataset on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

## 🔧 ML Pipeline

1. **Audio Preprocessing** - Load, resample (22,050 Hz), normalize
2. **Feature Extraction** - MFCC (13 coefficients)
3. **Dimensionality Reduction** - PCA (10 components, 99% variance)
4. **Scaling** - StandardScaler normalization
5. **Data Balancing** - SMOTE for class balance
6. **Model Training** - SVM with RBF kernel
7. **Prediction** - Emotion + confidence scores

## 🎯 Use Cases

- 📞 **Call Centers** - Customer sentiment analysis
- 🏥 **Healthcare** - Mental health monitoring
- 🤖 **AI Assistants** - Emotion-aware responses
- 📚 **Education** - Student engagement tracking
- 🎮 **Gaming** - Emotion-based interactions

## 🔒 Security Notes

- Models are loaded once at startup
- File size limited to 16MB
- File type validation (WAV, MP3, OGG, FLAC only)
- Temporary files cleaned after processing
- CORS configured for frontend integration

## 🚢 Deployment

### Backend (Heroku/AWS/GCP)
```bash
# Add Procfile
web: gunicorn app:app

# Deploy
git push heroku master
```

### Frontend (Netlify/Vercel)
```bash
cd frontend
vercel deploy
```

Update `frontend/script.js` with your backend URL:
```javascript
const API_BASE_URL = 'https://your-backend-url.com';
```

## 📝 Testing

**Test Pipeline:**
```bash
python test_pipeline.py
```

**API Test:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@path/to/audio.wav"
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- TESS Dataset creators
- Toronto Metropolitan University
- Open source community

---

**Built with ❤️ using Python, Flask, and Machine Learning**
