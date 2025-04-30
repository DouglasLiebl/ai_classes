import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
from PIL import Image
from argparse import ArgumentParser
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_neural_network_from_csv(
    csv_file: str,
    epochs: int = 1000,
    model_name: str = "model",
    layers: int = 4,
    neurons_by_layer: int = 4,
):
    """
    Trains a customizable neural network using the dataset provided in the CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        epochs (int): Number of training epochs. Default is 1000.
        layers (int): Number of hidden layers. Default is 3.
        neurons_per_layer (int): Number of neurons per hidden layer. Default is 4.

    Returns:
        tf.keras.Model: The trained neural network model.
    """

    dataset = pd.read_csv(csv_file)

    sns.countplot(x="class", data=dataset)

    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, -1].values
    # y = str(y).upper() == model_name.upper()
    
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
    
    # Add dropout for regularization
    model.add(tf.keras.layers.Dropout(0.2))

    for _ in range(layers - 1):
        print(f"Adding layer with {neurons_by_layer} neurons")
        model.add(tf.keras.layers.Dense(
            units=neurons_by_layer, 
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ))
        model.add(tf.keras.layers.Dropout(0.2))

    # Use appropriate output layer based on number of classes
    if unique_classes == 2:
        model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
        loss_function = "binary_crossentropy"
    else:
        model.add(tf.keras.layers.Dense(units=unique_classes, activation="softmax"))
        loss_function = "sparse_categorical_crossentropy"

    print(f"Using loss function: {loss_function}")
    model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])

    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )

    # Make training more verbose
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        validation_split=0.1, 
        verbose=1,
        callbacks=[early_stopping]
    )

    predictions = model.predict(X_test)
    predictions = predictions > 0.5
    acc = accuracy_score(y_test, predictions)
  

    models_dir = os.path.join("models")
    model_path = os.path.join(models_dir, f"{model_name}.keras")
    os.makedirs(models_dir, exist_ok=True)
    model.save(model_path)
    
    # Save class mapping
    class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    class_mapping_path = os.path.join(models_dir, f"{model_name}_classes.json")
    with open(class_mapping_path, "w") as f:
        json.dump(class_mapping, f)

    return {
        "model_name": model_name,
        "accuracy": float(str(acc)),
        "accuracy_str": str(acc),
    }


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


def extract_features(main_folder, output_csv, class_name, properties):
    header = [f"{prop.name}_{class_name}" for prop in properties]
    header.append("class")
    print(main_folder)

    rows = []

    for class_folder in os.listdir(main_folder):
        class_path = os.path.join(main_folder, class_folder)
        if not os.path.isdir(class_path):
            print(f"Skipping non-directory: {class_path}")
            continue

        for image_name in os.listdir(class_path):
            print(f"Processing image: {image_name} in class: {class_folder}")

            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path).convert("RGB")
            pixels = image.load()
            width, height = image.size
            total_pixels = width * height

            feature_counts = {f"{prop.name}_{class_name}": 0 for prop in properties}

            for x in range(width):
                for y in range(height):
                    pixel = pixels[x, y]
                    for prop in properties:
                        feature_key = f"{prop.name}_{class_name}"
                        interval = {"rgb": prop.rgb}
                        if is_pixel_in_interval(pixel, interval):
                            feature_counts[feature_key] += 1

            feature_proportions = [
                feature_counts[key] / total_pixels for key in header[:-1]
            ]
            feature_proportions.append(class_folder)
            rows.append(feature_proportions)

    print(
        f"Extracted {len(rows)} rows of features from {len(os.listdir(main_folder))} classes."
    )

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"CSV successfully generated: {output_csv}")

def predict_image(model_path, image_path, class_intervals_path):
    """
    Faz predição para uma única imagem.
    
    Args:
        model_path: Caminho para o modelo salvo
        image_path: Caminho para a imagem a ser classificada
        class_intervals_path: Caminho para o JSON com intervalos de classe
        
    Returns:
        dict: Predição com nome da classe e probabilidades
    """
    # Carrega modelo e mapeamentos
    model = tf.keras.models.load_model(model_path)
    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    class_mapping_path = os.path.join(model_dir, f"{model_name}_classes.json")
    
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)
    
    with open(class_intervals_path, "r") as f:
        class_intervals = json.load(f)
    
    # Extrai features da imagem (igual ao processo de treino)
    # Você precisará ter acesso à classe/estrutura Property usada no extract_features
    # Aqui assumimos que você tem uma função adaptada para uma única imagem
    features = extract_single_image_features(image_path, class_intervals)
    
    # Faz a predição
    prediction = model.predict(np.array([features]))
    
    # Processa a saída
    if len(class_mapping) == 2:  # Binário
        prob = prediction[0][0]
        class_idx = 1 if prob > 0.5 else 0
        probabilities = {class_mapping['0']: 1 - prob, class_mapping['1']: prob}
    else:  # Multiclasse
        class_idx = np.argmax(prediction)
        probabilities = {class_mapping[str(i)]: float(prediction[0][i]) 
                        for i in range(len(class_mapping))}
    
    return {
        'class': class_mapping[str(class_idx)],
        'probabilities': probabilities,
        'raw_prediction': prediction.tolist()
    }
def extract_single_image_features(image_path, class_intervals, properties=None):
    """
    Extrai features de uma única imagem no mesmo formato usado para treinar o modelo.
    
    Args:
        image_path (str): Caminho para a imagem a ser processada
        class_intervals (dict): Dicionário com os intervalos de classe (como no JSON)
        properties (list, optional): Lista de propriedades (como no treino). Se None, será criada.
        
    Returns:
        numpy.array: Array com as features extraídas (no mesmo formato do CSV de treino)
    """
    # Se properties não for fornecido, recriamos a estrutura usada no treino
    if properties is None:
        class Property:
            def __init__(self, name, rgb):
                self.name = name
                self.rgb = rgb
        
        properties = []
        for class_name, intervals in class_intervals.items():
            for interval_name, rgb_values in intervals.items():
                properties.append(Property(
                    name=f"{interval_name}_{class_name}",
                    rgb=rgb_values
                ))
    
    # Abre a imagem e converte para RGB
    image = Image.open(image_path).convert('RGB')
    pixels = image.load()
    width, height = image.size
    total_pixels = width * height
    
    # Inicializa contadores para cada feature
    feature_counts = {f"{prop.name}": 0 for prop in properties}
    
    # Processa cada pixel da imagem
    for x in range(width):
        for y in range(height):
            pixel = pixels[x, y]
            for prop in properties:
                interval = {"rgb": prop.rgb}
                if is_pixel_in_interval(pixel, interval):
                    feature_counts[prop.name] += 1
    
    # Calcula as proporções (igual ao processo de treino)
    feature_proportions = [count / total_pixels for count in feature_counts.values()]
    
    return np.array(feature_proportions)

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

    main_folder = args.main_folder
    output_csv = args.output_csv
    with open(args.class_intervals, "r") as f:
        class_intervals = json.load(f)

    # extract_features(main_folder, output_csv, class_intervals)
