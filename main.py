from fastapi import FastAPI, UploadFile, File, Form, Body, Request
from typing import List
import os
from datetime import datetime
import shutil
from service.cnn import train_and_evaluate_cnn, classify_new_image, list_trained_models
from service.feature_extraction import extract_features, train_neural_network_from_csv
from service.batch_upload_manager import BatchUploadManager
import tempfile
import traceback
import random
from entities.trainingParameters import TrainingParameters

app = FastAPI()


@app.post("/create-upload-session")
async def create_upload_session():
    try:
        session_id = BatchUploadManager.create_session()
        return {
            "status": "success",
            "session_id": session_id,
            "message": "Upload session created successfully. Use this session_id for batch uploads.",
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error creating upload session: {str(e)}\n{error_details}")
        return {"error": f"Failed to create upload session: {str(e)}"}


@app.post("/upload-batch/{session_id}")
async def upload_batch(
    session_id: str, class_name: str = Form(...), files: List[UploadFile] = File(...)
):
    try:

        try:
            BatchUploadManager.get_metadata(session_id)
        except ValueError:
            return {"error": f"Upload session {session_id} not found"}

        if len(files) > 50:
            return {"error": "Maximum batch size is 50 files"}

        print(f"Processing batch of {len(files)} files for class '{class_name}'")

        processed_files = []
        for file in files:
            content = await file.read()
            processed_files.append({"filename": file.filename, "content": content})

        updated_metadata = BatchUploadManager.add_files(
            session_id, class_name, processed_files
        )

        return {
            "status": "success",
            "message": f"Successfully uploaded {len(files)} files for class '{class_name}'",
            "session_id": session_id,
            "class_name": class_name,
            "files_in_class": updated_metadata["classes"][class_name]["count"],
            "total_files_in_session": updated_metadata["total_files"],
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Batch upload error: {str(e)}\n{error_details}")
        return {"error": f"Batch upload failed: {str(e)}"}


@app.get("/upload-sessions")
async def list_upload_sessions():
    try:
        sessions = BatchUploadManager.list_sessions()
        return {
            "status": "success",
            "sessions_count": len(sessions),
            "sessions": sessions,
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error listing sessions: {str(e)}\n{error_details}")
        return {"error": f"Failed to list sessions: {str(e)}"}


@app.post("/train-from-session/{session_id}")
async def train_from_session(session_id: str, params: TrainingParameters):
    try:
        try:
            metadata = BatchUploadManager.get_metadata(session_id)
        except ValueError:
            return {"error": f"Upload session {session_id} not found"}

        if metadata["total_files"] == 0:
            return {"error": "No files found in this upload session"}

        BatchUploadManager.update_metadata(session_id, {"status": "training"})

        parent_folder = "base"
        if os.path.exists(parent_folder):
            shutil.rmtree(parent_folder)

        os.makedirs(parent_folder, exist_ok=True)

        training_folder = os.path.join(parent_folder, "training_set")
        test_folder = os.path.join(parent_folder, "test_set")
        os.makedirs(training_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        test_ratio = params.test_percentage / 100.0
        if test_ratio < 0 or test_ratio > 1:
            return {"error": "Test percentage must be between 0 and 100"}

        saved_files = {"training_set": {}, "test_set": {}}
        session_dir = BatchUploadManager.get_session_path(session_id)

        for class_name in metadata["classes"].keys():
            class_dir = os.path.join(session_dir, class_name)

            file_paths = [
                os.path.join(class_dir, filename)
                for filename in os.listdir(class_dir)
                if os.path.isfile(os.path.join(class_dir, filename))
                and filename != "metadata.json"
            ]

            if not file_paths:
                continue

            train_class_folder = os.path.join(training_folder, class_name)
            test_class_folder = os.path.join(test_folder, class_name)
            os.makedirs(train_class_folder, exist_ok=True)
            os.makedirs(test_class_folder, exist_ok=True)

            saved_files["training_set"][class_name] = []
            saved_files["test_set"][class_name] = []

            random.shuffle(file_paths)

            test_count = max(1, int(len(file_paths) * test_ratio))
            train_count = len(file_paths) - test_count

            for i, file_path in enumerate(file_paths):
                is_test = i >= train_count
                target_folder = test_class_folder if is_test else train_class_folder
                set_type = "test_set" if is_test else "training_set"

                filename = os.path.basename(file_path)
                dest_path = os.path.join(target_folder, filename)
                shutil.copy2(file_path, dest_path)

                file_size = os.path.getsize(dest_path)
                saved_files[set_type][class_name].append(
                    {
                        "original_path": file_path,
                        "saved_path": dest_path,
                        "size": file_size,
                    }
                )

        if not saved_files["training_set"] or not saved_files["test_set"]:
            return {
                "error": "No valid files were uploaded or split between training and test sets"
            }

        if params.rgb_ranges is not None:
            try:
                features_csv = os.path.join(
                    parent_folder, params.model_name + "_features.csv"
                )
                extract_features(
                    main_folder=training_folder,
                    output_csv=features_csv,
                    class_name=params.model_name,
                    properties=params.rgb_ranges,
                )

                training_results = {}

                training_results = train_neural_network_from_csv(
                    features_csv,
                    epochs=params.epochs,
                    model_name=params.model_name,
                    layers=params.layers,
                    neurons_by_layer=params.neurons_by_layer,
                )

                return {
                    "message": "Files uploaded and model trained successfully",
                    "training_results": training_results,
                }

            except Exception as e:
                raise e
                return {"error": f"Request failed: {str(e)}"}

        training_results = train_and_evaluate_cnn(
            base_folder="./base",
            epochs=params.epochs,
            model_name=params.model_name or f"model_{session_id[:8]}",
            layers=params.layers,
            neurons_by_layer=params.neurons_by_layer,
        )

        BatchUploadManager.update_metadata(session_id, {"status": "completed"})

        try:
            shutil.rmtree(parent_folder)
        except Exception as e:
            print(f"Error deleting folder: {str(e)}")

        BatchUploadManager.clean_session(session_id, preserve_metadata=False)

        return {
            "status": "success",
            "message": "Model trained successfully",
            "session_id": session_id,
            "training_parameters": {
                "epochs": params.epochs,
                "model_name": params.model_name,
                "layers": params.layers,
                "neurons_by_layer": params.neurons_by_layer,
                "test_percentage": params.test_percentage,
            },
            "dataset_summary": {
                "classes": list(metadata["classes"].keys()),
                "files_per_class": {
                    cls: info["count"] for cls, info in metadata["classes"].items()
                },
                "total_files": metadata["total_files"],
                "training_files": {
                    class_name: len(files)
                    for class_name, files in saved_files["training_set"].items()
                },
                "test_files": {
                    class_name: len(files)
                    for class_name, files in saved_files["test_set"].items()
                },
            },
            "training_results": training_results,
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Training error: {str(e)}\n{error_details}")
        try:
            BatchUploadManager.update_metadata(session_id, {"status": "error"})
        except:
            pass

        return {"error": f"Training failed: {str(e)}"}


@app.post("/classify-image/")
async def classify_image(image: UploadFile = File(...), model_name: str = Form(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            contents = await image.read()
            temp_image.write(contents)
            temp_image_path = temp_image.name

        try:
            result = classify_new_image(model_name, temp_image_path)
            return {"model_name": model_name, "classification_result": result}

        finally:
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"Classification error: {str(e)}\n{error_details}")
        return {"error": f"Classification failed: {str(e)}"}


@app.get("/list-models/")
async def list_models():
    try:
        models = list_trained_models()
        return {"models_count": len(models), "models": models}
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error listing models: {str(e)}\n{error_details}")
        return {"error": f"Failed to list models: {str(e)}"}
