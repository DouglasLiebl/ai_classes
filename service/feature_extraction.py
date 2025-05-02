import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import os
import csv
from PIL import Image
from argparse import ArgumentParser
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from datetime import datetime
from typing import List, Dict
import glob

def train_neural_network_from_csv(
    csv_file: str,
    epochs: int = 1000,
    model_name: str = "model",
    layers: int = 4,
    neurons_by_layer: int = 4,
):
    dataset = pd.read_csv(csv_file)

    sns.countplot(x="class", data=dataset)

    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, -1].values

    unique_classes = len(np.unique(y))
    print(f"Number of unique classes: {unique_classes}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Dense(
            units=neurons_by_layer, activation="relu", input_shape=(X_train.shape[1],)
        )
    )

    model.add(tf.keras.layers.Dropout(0.2))

    for _ in range(layers - 1):
        print(f"Adding layer with {neurons_by_layer} neurons")
        model.add(
            tf.keras.layers.Dense(
                units=neurons_by_layer,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            )
        )
        model.add(tf.keras.layers.Dropout(0.2))

    if unique_classes == 2:
        model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
        loss_function = "binary_crossentropy"
    else:
        model.add(tf.keras.layers.Dense(units=unique_classes, activation="softmax"))
        loss_function = "sparse_categorical_crossentropy"

    print(f"Using loss function: {loss_function}")
    model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stopping],
    )

    predictions = model.predict(X_test)
    
    if unique_classes == 2:
        predictions_classes = (predictions > 0.5).astype(int).flatten()
    else:
        predictions_classes = np.argmax(predictions, axis=1)

    acc = accuracy_score(y_test, predictions_classes)

    models_dir = os.path.join("models", "rgb")
    model_path = os.path.join(models_dir, f"{model_name}.keras")
    os.makedirs(models_dir, exist_ok=True)
    model.save(model_path)

    class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    class_mapping_path = os.path.join(models_dir, f"{model_name}_classes.json")

    with open(class_mapping_path, "w") as f:
        json.dump(class_mapping, f)

    metadata = {
        "accuracy": float(str(acc)),
        "accuracy_str": str(acc),
        "created_date": datetime.now().isoformat(),
        "layers": layers,
        "neurons_by_layer": neurons_by_layer,
        "epochs": epochs,
        "training_samples": len(y_train),
        "test_samples": len(y_test),
        "unique_classes": unique_classes
    }
    
    metadata_path = os.path.join(models_dir, f"{model_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


def is_pixel_in_interval(pixel, interval):
    r, g, b = pixel
    r_default, g_default, b_default = interval["rgb"]

    tolerance = 10
    r_min, r_max = r_default - tolerance, r_default + tolerance
    g_min, g_max = g_default - tolerance, g_default + tolerance
    b_min, b_max = b_default - tolerance, b_default + tolerance

    r_min = max(0, r_min)
    r_max = min(255, r_max)

    g_min = max(0, g_min)
    g_max = min(255, g_max)

    b_min = max(0, b_min)
    b_max = min(255, b_max)

    return (r_min <= r <= r_max) and (g_min <= g <= g_max) and (b_min <= b <= b_max)


def extract_features(main_folder, output_csv, classes):
    header = []
    all_properties = []

    for training_class in classes:
        for prop in training_class.rgb_ranges:
            col_name = f"{prop.name}_{training_class.class_name}"
            if col_name not in header:
                header.append(col_name)
                all_properties.append((col_name, prop.rgb, training_class.class_name))

    rows = []

    for class_folder in os.listdir(main_folder):
        class_path = os.path.join(main_folder, class_folder)
        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):
            print(f"Processing image: {image_name} in class: {class_folder}")
            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path).convert("RGB")
            pixels = image.load()
            width, height = image.size
            total_pixels = width * height

            feature_counts = {col: 0 for col in header}

            for x in range(width):
                for y in range(height):
                    pixel = pixels[x, y]
                    for col_name, rgb, _ in all_properties:
                        if is_pixel_in_interval(pixel, {"rgb": rgb}):
                            feature_counts[col_name] += 1

            feature_proportions = [feature_counts[col] / total_pixels for col in header]
            feature_proportions.append(class_folder)
            rows.append(feature_proportions)

    header_with_class = header + ["class"]
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header_with_class)
        writer.writerows(rows)

    print(f"CSV successfully generated: {output_csv}")


def classify_feature_based_image(image_path, model_name, properties, class_name):
    model_path = os.path.join("models", "rgb", f"{model_name}.keras")
    model = load_model(model_path)

    image = Image.open(image_path).convert("RGB")
    pixels = image.load()
    width, height = image.size
    total_pixels = width * height

    feature_counts = {}
    for prop in properties:
        feature_counts[f"{prop.name}_{class_name}"] = 0

    for x in range(width):
        for y in range(height):
            pixel = pixels[x, y]
            for prop in properties:
                r, g, b = pixel
                r_ref, g_ref, b_ref = prop.rgb
                tolerance = 10
                if (
                    r_ref - tolerance <= r <= r_ref + tolerance
                    and g_ref - tolerance <= g <= g_ref + tolerance
                    and b_ref - tolerance <= b <= b_ref + tolerance
                ):
                    feature_counts[f"{prop.name}_{class_name}"] += 1

    feature_vector = [
        feature_counts[f"{prop.name}_{class_name}"] / total_pixels
        for prop in properties
    ]
    input_array = np.array([feature_vector])

    prediction = model.predict(input_array)
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class])

    class_mapping_path = os.path.join("models", "rgb", f"{model_name}_classes.json")
    class_name = str(predicted_class)

    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, "r") as f:
            class_mapping = json.load(f)
            class_name = class_mapping.get(str(predicted_class), class_name)

    return {
        "class_index": int(predicted_class),
        "class_name": class_name,
        "confidence": float(confidence),
        "all_probabilities": [float(p) for p in prediction[0]],
    }

def list_rgb_models() -> List[Dict]:
    models_dir = os.path.join("models", "rgb")
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
                with open(class_mapping_path, "r") as f:
                    classes = json.load(f)
            except:
                pass
        
        metadata = {
            "accuracy": None,
            "created_date": created_date.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        metadata_path = os.path.join(models_dir, f"{model_name}_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    stored_metadata = json.load(f)
                    metadata.update(stored_metadata)
                    if "created_date" in stored_metadata and isinstance(stored_metadata["created_date"], str):
                        metadata["created_date"] = stored_metadata["created_date"].split("T")[0] + " " + stored_metadata["created_date"].split("T")[1].split(".")[0]
            except Exception as e:
                print(f"Error loading metadata for {model_name}: {str(e)}")

        models_info.append(
            {
                "model_name": model_name,
                "model_size_mb": round(file_size / (1024 * 1024), 2),
                "created_date": metadata.get("created_date", created_date.strftime("%Y-%m-%d %H:%M:%S")),
                "num_classes": len(classes),
                "classes": list(classes.values()),
                "accuracy": metadata.get("accuracy"),
                "accuracy_str": metadata.get("accuracy_str", str(metadata.get("accuracy"))) if metadata.get("accuracy") is not None else None,
                "training_samples": metadata.get("training_samples"),
                "test_samples": metadata.get("test_samples"),
            }
        )

    models_info.sort(key=lambda x: x["created_date"], reverse=True)

    return models_info


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract pixel values from images.")

    parser.add_argument(
        "--class_intervals",
        type=str,
        default="class_intervals.json",
        help="Path to the JSON file containing class intervals.",
    )

    parser.add_argument(
        "--main_folder",
        type=str,
        default="images",
        help="Main folder containing image classes.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="pixels.csv",
        help="Output CSV file name.",
    )

    args = parser.parse_args()
