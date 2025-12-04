"""
Flask REST API for Speech Emotion Recognition
Provides endpoints for emotion prediction from audio files
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pipeline import SpeechEmotionPipeline
import os
from werkzeug.utils import secure_filename
import traceback

# Fix numba compilation issues on serverless platforms
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp'

# Initialize Flask app
app = Flask(__name__)

# Configure CORS properly
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://speech-emotion-recognition-woad.vercel.app",
            "http://localhost:8000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
MODELS_DIR = 'models'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained pipeline
print("Loading trained pipeline...")
pipeline = SpeechEmotionPipeline()
try:
    pipeline.load_pipeline(save_dir=MODELS_DIR)
    print("[OK] Pipeline loaded successfully!")
except Exception as e:
    print(f"[WARNING] Could not load pipeline. Error: {e}")
    print("Please train the model first by running: python train_model.py")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Speech Emotion Recognition API',
        'version': '1.0',
        'endpoints': {
            '/': 'API information (this page)',
            '/health': 'Health check',
            '/predict': 'Predict emotion from audio file (POST)',
            '/emotions': 'List of available emotions'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    is_ready = pipeline.best_model is not None
    return jsonify({
        'status': 'healthy' if is_ready else 'not_ready',
        'model_loaded': is_ready,
        'model_name': pipeline.best_model_name if is_ready else None
    })


@app.route('/emotions', methods=['GET'])
def emotions():
    """Get list of available emotions"""
    if pipeline.label_encoder is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 503

    return jsonify({
        'emotions': pipeline.label_encoder.classes_.tolist(),
        'count': len(pipeline.label_encoder.classes_)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict emotion from uploaded audio file

    Request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: file (audio file)

    Response:
        {
            "success": true,
            "predicted_emotion": "Happy",
            "confidence_scores": {
                "Angry": 0.05,
                "Happy": 0.85,
                ...
            },
            "model_used": "SVM"
        }
    """
    # Check if model is loaded
    if pipeline.best_model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first by running: python train_model.py'
        }), 503

    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided. Please upload an audio file.'
        }), 400

    file = request.files['file']

    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected.'
        }), 400

    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        result = pipeline.predict(filepath)

        # Clean up uploaded file
        os.remove(filepath)

        # Return result
        return jsonify({
            'success': True,
            'predicted_emotion': result['predicted_emotion'],
            'confidence_scores': result['confidence_scores'],
            'model_used': result['model_used'],
            'filename': filename
        })

    except Exception as e:
        # Clean up file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)

        # Return error
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error occurred.'
    }), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
