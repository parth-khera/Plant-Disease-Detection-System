from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
from werkzeug.utils import secure_filename
from PIL import Image
import io
from llm_api import ask_gemini, ask_perplexity

app = Flask(__name__)
app.config['SECRET_KEY'] = 'plant-disease-detection-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global variables for model and class names
model = None
class_names = None
treatment_db = None

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_treatment_database():
    """Load treatment suggestions database"""
    global treatment_db
    try:
        with open('treatment_suggestions.json', 'r') as f:
            treatment_db = json.load(f)
    except FileNotFoundError:
        # Create default treatment database
        treatment_db = {
            "healthy": {
                "description": "Your plant appears to be healthy!",
                "treatment": "Continue providing proper care: adequate watering, sunlight, and nutrients.",
                "prevention": "Maintain good plant care practices to prevent diseases."
            },
            "bacterial_spot": {
                "description": "Bacterial spot disease affects leaves with small, dark spots.",
                "treatment": "Remove affected leaves immediately. Apply copper-based fungicide. Improve air circulation.",
                "prevention": "Avoid overhead watering. Space plants properly for air circulation."
            },
            "early_blight": {
                "description": "Early blight causes dark spots with concentric rings on leaves.",
                "treatment": "Remove affected leaves. Apply fungicide containing chlorothalonil or mancozeb.",
                "prevention": "Rotate crops annually. Mulch around plants to prevent soil splash."
            },
            "late_blight": {
                "description": "Late blight causes water-soaked lesions that turn brown and fuzzy.",
                "treatment": "Remove and destroy affected plants immediately. Apply fungicide promptly.",
                "prevention": "Plant resistant varieties. Avoid working with wet plants."
            }
        }
        # Save the default database
        with open('treatment_suggestions.json', 'w') as f:
            json.dump(treatment_db, f, indent=4)

def load_ml_model():
    """Load the trained machine learning model"""
    global model, class_names
    try:
        model_path = 'model/plant_disease_model.h5'
        if os.path.exists(model_path):
            model = load_model(model_path)
            print("Model loaded successfully!")

            # Load class names
            class_names_path = 'model/class_names.npy'
            if os.path.exists(class_names_path):
                class_names = np.load(class_names_path, allow_pickle=True)
                print(f"Class names loaded: {class_names}")
            else:
                # Default class names if file not found
                class_names = ['healthy', 'bacterial_spot', 'early_blight', 'late_blight']
                print("Using default class names")
        else:
            print(f"Model file not found at {model_path}")
            print("Please train the model first using: python train_model.py")
            model = None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

def preprocess_image(img_path):
    """Preprocess image for model prediction"""
    try:
        # Load and resize image
        img = image.load_img(img_path, target_size=(224, 224))

        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_disease(img_path):
    """Predict plant disease from image"""
    if model is None:
        return None, "Model not loaded. Please train the model first."

    try:
        # Preprocess image
        processed_img = preprocess_image(img_path)
        if processed_img is None:
            return None, "Error processing image"

        # Make prediction
        predictions = model.predict(processed_img)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        # Get predicted class name
        if isinstance(class_names, np.ndarray):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = class_names[predicted_class_idx]

        return predicted_class, confidence

    except Exception as e:
        return None, f"Prediction error: {str(e)}"

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Secure filename
        filename = secure_filename(file.filename)

        # Create uploads directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Save file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Make prediction
        predicted_class, confidence = predict_disease(file_path)

        if predicted_class is None:
            flash(f'Prediction failed: {confidence}')
            return redirect(url_for('index'))

        # Get treatment information
        treatment_info = treatment_db.get(predicted_class, {
            "description": "Disease information not available",
            "treatment": "Please consult a plant specialist",
            "prevention": "Maintain proper plant care"
        })

        # Get LLM feedback
        try:
            llm_feedback = ask_gemini(f"Provide detailed advice and additional tips for treating {predicted_class.replace('_', ' ')} disease in plants. Include preventive measures and any home remedies if applicable.")
        except Exception as e:
            llm_feedback = "Additional AI insights are currently unavailable. Please refer to the treatment recommendations above."

        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass

        # Render results
        return render_template('results.html',
                              prediction=predicted_class,
                              confidence=f"{confidence * 100:.2f}",
                              treatment_info=treatment_info,
                              llm_feedback=llm_feedback,
                              filename=filename)

    else:
        flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP)')
        return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    flash('Internal server error. Please try again.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Load treatment database
    load_treatment_database()

    # Load ML model
    load_ml_model()

    # Create uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)