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
from sklearn.metrics import confusion_matrix

def train_and_evaluate_cnn(base_folder='./base', epochs=100, save_model=True):
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
    
    neural_network.add(Flatten())
    
    neural_network.add(Dense(units=6, activation='relu'))
    neural_network.add(Dense(units=4, activation='relu'))
    neural_network.add(Dense(units=4, activation='relu'))
    neural_network.add(Dense(units=4, activation='relu'))
    neural_network.add(Dense(units=len(training_base.class_indices), activation='softmax'))
    
    neural_network.compile(
        optimizer='adam', 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = neural_network.fit(
        training_base,
        steps_per_epoch=len(training_base),
        epochs=epochs,
        validation_data=test_base,
        validation_steps=len(test_base)
    )
    
    predictions = neural_network.predict(test_base)
    predictions_classes = np.argmax(predictions, axis=1)
    acc = accuracy_score(predictions_classes, test_base.classes)

    cm = confusion_matrix(predictions_classes, test_base.classes)
    
    if save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"cnn_model_{timestamp}.h5"
        neural_network.save(model_path)
    else:
        model_path = None
    
    accuracy_values = [float(str(x)) for x in history.history['accuracy']]
    val_accuracy_values = [float(str(x)) for x in history.history['val_accuracy']]
    loss_values = [float(str(x)) for x in history.history['loss']]
    val_loss_values = [float(str(x)) for x in history.history['val_loss']]
    
    return {
        "accuracy": float(str(acc)),
        "accuracy_str": str(acc),
        "model_path": model_path,
        "class_indices": test_base.class_indices,
        "confusion_matrix": cm.tolist(),
        "history": {
            "accuracy": accuracy_values,
            "val_accuracy": val_accuracy_values,
            "loss": loss_values,
            "val_loss": val_loss_values
        }
    }