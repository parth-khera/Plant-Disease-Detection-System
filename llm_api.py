import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import requests
import shutil
import google.generativeai as genai
from config import GEMINI_API_KEY, PERPLEXITY_API_KEY

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("⚠️  kagglehub not installed. Run: pip install kagglehub")

def create_model(num_classes=4):
    """
    Create a plant disease detection model using MobileNetV2 transfer learning

    Args:
        num_classes: Number of disease classes to classify

    Returns:
        Compiled Keras model
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def create_data_generators(train_dir, val_dir=None, test_dir=None, batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    val_generator = None
    if val_dir and os.path.exists(val_dir):
        val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    test_generator = None
    if test_dir and os.path.exists(test_dir):
        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    return train_generator, val_generator, test_generator

def download_and_organize_dataset():
    """
    Download PlantVillage dataset using kagglehub and organize it for training
    """
    if not KAGGLEHUB_AVAILABLE:
        print("❌ kagglehub not available. Please install it: pip install kagglehub")
        return False

    try:
        print("📥 Downloading PlantVillage dataset from Kaggle...")
        # Download the dataset
        dataset_path = kagglehub.dataset_download("emmarex/plantdisease")
        print(f"✅ Dataset downloaded to: {dataset_path}")

        # The dataset structure from kagglehub might be different
        # Let's explore what we got
        print("📂 Exploring dataset structure...")
        for root, dirs, files in os.walk(dataset_path):
            level = root.replace(dataset_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # Show only first 3 files per directory
                print(f"{subindent}{file}")
            if len(files) > 3:
                print(f"{subindent}... and {len(files) - 3} more files")

        # Now we need to organize it into our expected structure
        # The PlantVillage dataset typically has a structure like:
        # dataset_path/
        #   ├── PlantVillage/
        #       ├── Potato___healthy/
        #       ├── Potato___Early_blight/
        #       ├── Tomato___Bacterial_spot/
        #       └── etc.

        data_dir = 'data'
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'validation')

        # Create directories
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Map PlantVillage class names to our expected classes
        class_mapping = {
            'healthy': ['healthy', 'Healthy'],
            'bacterial_spot': ['Bacterial_spot', 'Bacterial_Spot'],
            'early_blight': ['Early_blight', 'Early_Blight'],
            'late_blight': ['Late_blight', 'Late_Blight']
        }

        print("🔄 Organizing dataset into training/validation structure...")

        # Find the actual dataset directory (it might be nested)
        plantvillage_dir = None
        for root, dirs, files in os.walk(dataset_path):
            if 'PlantVillage' in dirs:
                plantvillage_dir = os.path.join(root, 'PlantVillage')
                break
            elif 'PlantVillage' in root:
                plantvillage_dir = root
                break

        if not plantvillage_dir or not os.path.exists(plantvillage_dir):
            print("❌ Could not find PlantVillage directory in downloaded dataset")
            return False

        print(f"📁 Found PlantVillage data at: {plantvillage_dir}")

        # Process each class
        for our_class, pv_classes in class_mapping.items():
            train_class_dir = os.path.join(train_dir, our_class)
            val_class_dir = os.path.join(val_dir, our_class)

            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            images_found = 0

            # Look for matching directories in the PlantVillage dataset
            for root, dirs, files in os.walk(plantvillage_dir):
                dir_name = os.path.basename(root)

                # Check if this directory matches any of our target classes
                for pv_class in pv_classes:
                    if pv_class.lower() in dir_name.lower():
                        print(f"  📋 Processing {dir_name} -> {our_class}")

                        # Get all image files
                        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

                        if not image_files:
                            continue

                        # Split 80/20 for train/validation
                        split_idx = int(len(image_files) * 0.8)
                        train_files = image_files[:split_idx]
                        val_files = image_files[split_idx:]

                        # Copy training files
                        for file in train_files:
                            src = os.path.join(root, file)
                            dst = os.path.join(train_class_dir, f"{dir_name}_{file}")
                            shutil.copy2(src, dst)

                        # Copy validation files
                        for file in val_files:
                            src = os.path.join(root, file)
                            dst = os.path.join(val_class_dir, f"{dir_name}_{file}")
                            shutil.copy2(src, dst)

                        images_found += len(image_files)
                        print(f"    ✅ Copied {len(train_files)} train, {len(val_files)} validation images")

            if images_found == 0:
                print(f"  ⚠️  No images found for class: {our_class}")

        print("✅ Dataset organization complete!")
        return True

    except Exception as e:
        print(f"❌ Error downloading/organizing dataset: {str(e)}")
        return False

def train_model():
    class_names = ['healthy', 'bacterial_spot', 'early_blight', 'late_blight']
    num_classes = len(class_names)
    model = create_model(num_classes)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model Summary:")
    model.summary()
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    # Check if data directories exist, create them if they don't
    if not os.path.exists(data_dir):
        print(f"Creating data directory: {data_dir}")
        os.makedirs(data_dir)

    if not os.path.exists(train_dir):
        print(f"Creating training data directory: {train_dir}")
        os.makedirs(train_dir)

    if not os.path.exists(val_dir):
        print(f"Creating validation data directory: {val_dir}")
        os.makedirs(val_dir)

    # Create class subdirectories
    for class_name in class_names:
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)

        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
            print(f"Created: {train_class_dir}")

        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)
            print(f"Created: {val_class_dir}")

    # Check if there are any images in the training directory
    total_images = 0
    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            total_images += len(images)
            print(f"Found {len(images)} images in {class_name}/")

    if total_images == 0:
        print("\n⚠️  No training images found!")
        print("🔄 Attempting to download PlantVillage dataset automatically...")

        # Try to download the dataset automatically
        if download_and_organize_dataset():
            print("✅ Dataset downloaded and organized successfully!")
            print("🔄 Re-checking for images...")

            # Re-check for images after download
            total_images = 0
            for class_name in class_names:
                class_dir = os.path.join(train_dir, class_name)
                if os.path.exists(class_dir):
                    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                    total_images += len(images)
                    print(f"Found {len(images)} images in {class_name}/")

            if total_images == 0:
                print("\n❌ Still no images found after download attempt.")
                print("Please check your internet connection and try again.")
                return None, None, None
        else:
            print("\n❌ Automatic download failed.")
            print("Please manually add plant leaf images to the following directories:")
            print("data/train/healthy/ - images of healthy leaves")
            print("data/train/bacterial_spot/ - images of leaves with bacterial spot")
            print("data/train/early_blight/ - images of leaves with early blight")
            print("data/train/late_blight/ - images of leaves with late blight")
            print("data/validation/ - same structure for validation data")
            print("\n💡 Tip: You can download plant disease datasets from:")
            print("   - PlantVillage Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease")
            print("   - Or search for 'plant disease detection dataset' on Kaggle")
            return None, None, None
    train_generator, val_generator, _ = create_data_generators(
        train_dir, val_dir, batch_size=32
    )
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy' if val_generator else 'accuracy',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'model/plant_disease_model.h5',
            monitor='val_accuracy' if val_generator else 'accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator) if val_generator else None
    print("Starting model training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    model.save('model/plant_disease_model.h5')
    print("Model saved as 'model/plant_disease_model.h5'")
    np.save('model/class_names.npy', class_names)
    print("Class names saved as 'model/class_names.npy'")

    # LLM feedback section
    final_acc = history.history['accuracy'][-1]
    val_acc = history.history.get('val_accuracy', [None])[-1] if 'val_accuracy' in history.history else None
    summary = f"Model trained on plant disease detection. Final training accuracy: {final_acc:.2f}"
    if val_acc:
        summary += f", Validation accuracy: {val_acc:.2f}"
    print("\n📡 Asking Gemini for training insights...")
    gemini_prompt = f"{summary}. Provide detailed suggestions for improving this plant disease detection model, including data augmentation techniques, hyperparameter tuning, and architectural changes."
    gemini_response = ask_gemini(gemini_prompt)
    print("Gemini says:\n", gemini_response)
    print("\n📡 Asking Perplexity for evaluation...")
    perplexity_prompt = f"{summary}. As an AI expert, suggest advanced optimizations, regularization techniques, and best practices for this CNN model training on plant disease classification."
    perplexity_response = ask_perplexity(perplexity_prompt)
    print("Perplexity says:\n", perplexity_response)

    # Save LLM feedback to a file
    with open('model/training_feedback.txt', 'w') as f:
        f.write("=== TRAINING SUMMARY ===\n")
        f.write(f"{summary}\n\n")
        f.write("=== GEMINI FEEDBACK ===\n")
        f.write(f"{gemini_response}\n\n")
        f.write("=== PERPLEXITY FEEDBACK ===\n")
        f.write(f"{perplexity_response}\n")
    print("LLM feedback saved to model/training_feedback.txt")

    return model, history, class_names

def ask_gemini(prompt):
    """Ask Gemini AI for a response"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error with Gemini API: {str(e)}"

def ask_perplexity(prompt):
    """Ask Perplexity AI for a response"""
    try:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error with Perplexity API: {str(e)}"

if __name__ == "__main__":
    result = train_model()

    # Check if training was successful
    if result[0] is None:
        print("\n❌ Training cancelled due to missing data.")
        print("Please add training images and try again.")
        exit(1)

    model, history, class_names = result

    print("\nTraining completed!")
    print(f"Model saved to: model/plant_disease_model.h5")
    print(f"Class names: {class_names}")