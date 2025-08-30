# Plant Disease Detection System

A web-based application that uses machine learning to detect plant diseases from leaf images. Built with TensorFlow/Keras and Flask.

## Features

- **AI-Powered Disease Detection**: Uses MobileNetV2 transfer learning model to classify plant diseases
- **Web Interface**: Clean, responsive web application for easy image upload and analysis
- **Treatment Recommendations**: Provides treatment suggestions and prevention tips for detected diseases
- **Confidence Scoring**: Shows prediction confidence percentage
- **LLM-Powered Insights**: Integrated Gemini and Perplexity AI for enhanced feedback and training optimization
- **AI Training Assistance**: LLMs provide suggestions for model improvement during and after training
- **Multiple Disease Support**: Currently supports detection of:
  - Healthy plants
  - Bacterial Spot
  - Early Blight
  - Late Blight

## Project Structure

```
plant-disease-detection/
├── app.py                          # Flask web application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── treatment_suggestions.json      # Disease treatment database
├── templates/
│   ├── index.html                  # Upload page
│   └── results.html                # Results display page
├── static/
│   └── style.css                   # CSS styling
├── model/                          # Trained model storage
├── uploads/                        # Temporary uploaded files
└── data/                           # Training data (to be created)
    ├── train/
    └── validation/
```

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys** (for LLM features):
   - Get Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Get Perplexity API key from [Perplexity AI](https://www.perplexity.ai/settings/api)
   - Update `config.py` with your API keys

4. **Prepare training data** (optional - directories are created automatically):
   ```
   data/
   ├── train/
   │   ├── healthy/          # Images of healthy leaves
   │   ├── bacterial_spot/   # Images with bacterial spot
   │   ├── early_blight/     # Images with early blight
   │   └── late_blight/      # Images with late blight
   └── validation/           # Same structure for validation data
   ```

   **Dataset Sources:**
   - [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) on Kaggle
   - Search for "plant disease detection dataset" on academic repositories
   - Collect your own images using a smartphone camera

## Usage

### Training the Model

1. **Prepare your dataset** in the structure shown above
2. **Run the training script**:
   ```bash
   python train_model.py
   ```
3. The trained model will be saved as `model/plant_disease_model.h5`

### Running the Web Application

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and go to:
   ```
   http://127.0.0.1:5000
   ```

3. **Upload a plant leaf image** and click "Analyze Disease"

4. **View results** including:
   - Disease prediction
   - Confidence score
   - Treatment recommendations
   - Prevention tips
   - **AI-Powered Insights** (from Gemini LLM for additional advice)

## Model Details

- **Architecture**: MobileNetV2 with custom classification head
- **Input Size**: 224x224 pixels
- **Preprocessing**: Image normalization and augmentation
- **Transfer Learning**: Pre-trained on ImageNet
- **Fine-tuning**: Last layers trained on plant disease dataset

## LLM Integration

The system integrates two AI models for enhanced functionality:

### Gemini AI (Google)
- **Purpose**: Provides detailed treatment advice and user feedback
- **Usage**: Called during prediction to give personalized recommendations
- **Features**: Natural language explanations, home remedies, preventive measures

### Perplexity AI
- **Purpose**: Expert evaluation and training optimization
- **Usage**: Called after training to analyze performance and suggest improvements
- **Features**: Advanced technical suggestions, hyperparameter tuning, architectural changes

### LLM Features
- **Training Feedback**: After model training, both LLMs analyze performance and provide detailed improvement suggestions
- **User Insights**: For each prediction, Gemini provides additional context and advice
- **Feedback Storage**: All LLM responses are saved to `model/training_feedback.txt`
- **Error Handling**: Graceful fallback when API keys are missing or services unavailable

### Setup Requirements
1. **Gemini API Key**: Obtain from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Perplexity API Key**: Obtain from [Perplexity Settings](https://www.perplexity.ai/settings/api)
3. **Configuration**: Update `config.py` with your API keys

## Supported File Formats

- JPEG/JPG
- PNG
- GIF
- BMP
- Maximum file size: 16MB

## API Endpoints

- `GET /` - Home page with upload form
- `POST /predict` - Image upload and prediction endpoint
- `GET /about` - About page (can be added)

## Customization

### Adding New Diseases

1. **Update the class names** in `train_model.py`:
   ```python
   class_names = ['healthy', 'bacterial_spot', 'early_blight', 'late_blight', 'new_disease']
   ```

2. **Add treatment information** in `treatment_suggestions.json`:
   ```json
   "new_disease": {
     "description": "Description of the new disease",
     "treatment": "Treatment recommendations",
     "prevention": "Prevention tips"
   }
   ```

3. **Update training data** with images of the new disease category

### Modifying the Model

- Change the base model in `create_model()` function
- Adjust preprocessing parameters
- Modify training hyperparameters

## Troubleshooting

### Model Not Loading
- Ensure `model/plant_disease_model.h5` exists
- Check that all required packages are installed
- Verify TensorFlow/Keras compatibility

### Upload Issues
- Check file size (max 16MB)
- Verify supported image formats
- Ensure proper permissions for uploads directory

### Training Issues
- **Directory Creation**: The system automatically creates the required data directories
- **No Images Found**: If no training images are found, helpful links to datasets are provided
- **Dataset Structure**: Ensure images are placed in the correct class subdirectories
- **Check image quality and quantity**: Minimum 10-20 images per class recommended
- **Ensure sufficient computational resources**: Training requires GPU for faster processing

### LLM Issues
- **API Keys Missing**: Update `config.py` with valid Gemini and Perplexity API keys
- **Network Issues**: Ensure internet connection for LLM API calls
- **Rate Limits**: Free API tiers have usage limits; consider upgrading for heavy usage
- **Fallback Mode**: System continues to work without LLM features if APIs are unavailable

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Flask 2.3+
- Pillow 10.0+
- NumPy 1.24+
- Requests 2.31+
- Google Generative AI 0.3+
- **API Keys Required**:
  - Gemini API key (from Google AI Studio)
  - Perplexity API key (from Perplexity AI)

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Disclaimer

This tool is for educational and informational purposes only. It should not replace professional agricultural advice or laboratory testing. Always consult with certified plant pathologists for accurate disease diagnosis and treatment recommendations.