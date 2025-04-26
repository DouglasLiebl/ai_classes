import os
from datetime import datetime
from fastapi import Request
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.preprocessing import image
import json
from typing import List, Dict
import glob

def train_and_evaluate_cnn(
    base_folder='./base', 
    epochs=30, 
    save_model=True,
    model_name=datetime.now().strftime("%Y%m%d_%H%M%S"),
    layers=6,
    neurons_by_layer=6,
):
    training_folder = os.path.join(base_folder, 'training_set')
    test_folder = os.path.join(base_folder, 'test_set')
    
    if not os.path.exists(training_folder) or not os.path.exists(test_folder):
        return {
            "error": "Training or test folder not found",
            "training_folder": training_folder,
            "test_folder": test_folder
        }
        
    training_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=7,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    training_base = training_generator.flow_from_directory(
        training_folder,
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical'
    )
    
    test_generator = ImageDataGenerator(rescale=1./255)
    test_base = test_generator.flow_from_directory(
        test_folder,
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical',
        shuffle=False
    )
    
    neural_network = Sequential()
    neural_network.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
    neural_network.add(MaxPooling2D(pool_size=(2,2)))
    
    neural_network.add(Conv2D(32, (3,3), activation='relu'))
    neural_network.add(MaxPooling2D(pool_size=(2,2)))

    neural_network.add(Conv2D(32, (3,3), activation='relu'))
    neural_network.add(MaxPooling2D(pool_size=(2,2)))
    
    neural_network.add(Flatten())
    
    for _ in range(0, layers):
        neural_network.add(Dense(units=neurons_by_layer, activation='relu'))

    neural_network.add(Dense(units=len(training_base.class_indices), activation='softmax'))
    
    neural_network.compile(
        optimizer='adam', 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    neural_network.fit(
        training_base,
        steps_per_epoch=len(training_base),
        epochs=epochs,
        validation_data=test_base,
        validation_steps=len(test_base)
    )
    
    predictions = neural_network.predict(test_base)
    predictions_classes = np.argmax(predictions, axis=1)
    acc = accuracy_score(predictions_classes, test_base.classes)
    
    models_dir = os.path.join("models", "cnn")
    os.makedirs(models_dir, exist_ok=True)

    if save_model:
        model_path = os.path.join(models_dir, f"{model_name}.keras")
        neural_network.save(model_path)
        
        class_indices = training_base.class_indices
        class_mapping = {str(v): k for k, v in class_indices.items()}
        class_mapping_path = os.path.join(models_dir, f"{model_name}_classes.json")
        with open(class_mapping_path, 'w') as f:
            json.dump(class_mapping, f)
    
    return {
        "model_name": model_name,
        "accuracy": float(str(acc)),
        "accuracy_str": str(acc),
        "class_indices": test_base.class_indices,
    }

def classify_new_image(model_name, image_path):
    try:
        models_dir = os.path.join("models", "cnn")
        model_path = os.path.join(models_dir, f"{model_name}.keras")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = load_model(model_path)
        
        img = image.load_img(image_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][predicted_class])
        
        class_name = str(predicted_class)
        class_mapping_path = os.path.join(models_dir, f"{model_name}_classes.json")
        if os.path.exists(class_mapping_path):
            try:
                with open(class_mapping_path, 'r') as f:
                    class_mapping = json.load(f)
                    class_name = class_mapping.get(str(predicted_class), class_name)
            except:
                pass
        
        return {
            "class_index": int(predicted_class),
            "class_name": class_name,
            "confidence": float(confidence),
            "all_probabilities": [float(p) for p in prediction[0]]
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in classify_new_image: {str(e)}\n{error_details}")
        raise

def list_trained_models() -> List[Dict]:
    models_dir = os.path.join("models", "cnn")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        return []
    
    model_paths = glob.glob(os.path.join(models_dir, "*.keras"))
    models_info = []
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace(".keras", "")
        
        stats = os.stat(model_path)
        created_date = datetime.fromtimestamp(stats.st_ctime)
        file_size = stats.st_size
        
        class_mapping_path = os.path.join(models_dir, f"{model_name}_classes.json")
        classes = {}
        if os.path.exists(class_mapping_path):
            try:
                with open(class_mapping_path, 'r') as f:
                    classes = json.load(f)
            except:
                pass
        
        models_info.append({
            "model_name": model_name,
            "model_size_mb": round(file_size / (1024 * 1024), 2),
            "created_date": created_date.strftime("%Y-%m-%d %H:%M:%S"),
            "num_classes": len(classes),
            "classes": list(classes.values())
        })
    
    models_info.sort(key=lambda x: x["created_date"], reverse=True)
    
    return models_info