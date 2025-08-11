import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import cv2
import gdown

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'model/best_model.keras'

# Google Drive URLs - Update these with your actual URLs
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=10Y8JsMi6GjiYNm9s1c2zzEXGH02mIFtC"
CLASS_NAMES_URL = "https://drive.google.com/uc?id=1yrWuqmKesWPIhxbEzX07lida-BbW9-QF"
COLAB_NOTEBOOK_URL = "https://colab.research.google.com/drive/14viVNyKRsvyJyntQhmnOaAyaY0IjOthh?usp=sharing"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)

# Global variables
model = None
CLASS_NAMES = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_model():
    """Download the model from Google Drive using gdown."""
    model_path = MODEL_PATH
    print(f"Downloading model from Google Drive...")
    try:
        gdown.download(MODEL_DRIVE_URL, model_path, quiet=False)
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            print(f"Model downloaded successfully to {model_path}")
            return model_path
        else:
            raise Exception("Downloaded file is empty or doesn't exist")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise e

def download_class_names():
    """Download the class names file from Google Drive."""
    class_names_path = 'model/class_names.txt'
    print(f"Downloading class names from Google Drive...")
    
    try:
        gdown.download(CLASS_NAMES_URL, class_names_path, quiet=False)
        
        if os.path.exists(class_names_path) and os.path.getsize(class_names_path) > 0:
            with open(class_names_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines()]
            print(f"Found {len(class_names)} classes")
            return class_names
        else:
            raise Exception("Downloaded class names file is empty or doesn't exist")
    
    except Exception as e:
        print(f"Error downloading class names: {e}")
        raise e

def load_class_names():
    try:
        with open('model/class_names.txt', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Class names file not found. Downloading...")
        try:
            return download_class_names()
        except Exception as e:
            print(f"Error downloading class names: {e}")
            # Return default class names as fallback
            return [f"class_{i}" for i in range(80)]  # Assuming 80 classes for Indian food

def load_model():
    try:
        # Check if model exists, if not download it
        if not os.path.exists(MODEL_PATH):
            print("Model not found. Downloading...")
            download_model()
        
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize model and class names
print("Initializing application...")
CLASS_NAMES = load_class_names()
model = load_model()

# Check if model is loaded
if model is None:
    print("Warning: Model could not be loaded. Predictions will not work.")
    
if not CLASS_NAMES:
    print("Warning: No class names found. Predictions will not be labeled correctly.")
else:
    print(f"✅ Loaded {len(CLASS_NAMES)} class names")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({'error': 'Unable to read the image file. Please upload a valid image.'}), 400
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            prediction = model.predict(img)
            
            # Get top 5 predictions for better user experience
            top_5_indices = np.argsort(prediction[0])[-5:][::-1]
            
            predictions = []
            for idx in top_5_indices:
                if idx < len(CLASS_NAMES):
                    predictions.append({
                        'class': CLASS_NAMES[idx],
                        'confidence': float(prediction[0][idx] * 100)
                    })
            
            # Main prediction (highest confidence)
            predicted_class_idx = top_5_indices[0]
            if predicted_class_idx >= len(CLASS_NAMES):
                predicted_class = f"Class_{predicted_class_idx}"
            else:
                predicted_class = CLASS_NAMES[predicted_class_idx]
            
            confidence = float(np.max(prediction) * 100)
            
            return jsonify({
                'class': predicted_class,
                'confidence': confidence,
                'image_path': f"/static/uploads/{filename}",
                'top_predictions': predictions[:3],  # Return top 3
                'debug_info': {
                    'total_classes': len(CLASS_NAMES),
                    'prediction_sum': float(prediction.sum()),
                    'model_confident': confidence > 50.0
                }
            })
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'})

@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'class_names_loaded': len(CLASS_NAMES) > 0,
        'total_classes': len(CLASS_NAMES)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
