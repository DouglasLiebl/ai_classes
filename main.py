from fastapi import FastAPI, UploadFile, File, Form
import os
from datetime import datetime
from fastapi import Request
import unicodedata
import shutil
from service.cnn import train_and_evaluate_cnn, classify_new_image, list_trained_models
import tempfile
import traceback

app = FastAPI()

def sanitize_filename(filename: str) -> str:
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    filename = filename.replace(' ', '_')
    return filename

@app.post("/cnn/train")
async def upload_images(
    request: Request, 
    epochs: int = 30, 
    model_name: str = "",
    layers: int = 6,
    neurons_by_layer: int = 6,
):
    parent_folder = "base"
    
    if os.path.exists(parent_folder):
        shutil.rmtree(parent_folder)
    
    os.makedirs(parent_folder, exist_ok=True)
    
    training_folder = os.path.join(parent_folder, "training_set")
    test_folder = os.path.join(parent_folder, "test_set")
    os.makedirs(training_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    try:
        form_data = await request.form()
        saved_files = {"training_set": {}, "test_set": {}}

        for field_name, file in form_data.multi_items():
            
            if field_name.startswith("training_"):
                class_name = field_name[len("training_"):]
                base_folder = training_folder
                set_type = "training_set"
            elif field_name.startswith("test_"):
                class_name = field_name[len("test_"):]
                base_folder = test_folder
                set_type = "test_set"
            else:
                class_name = field_name
                base_folder = parent_folder
                set_type = "other"
            
            class_folder = os.path.join(base_folder, class_name)
            if set_type in ["training_set", "test_set"]:
                if class_name not in saved_files[set_type]:
                    os.makedirs(class_folder, exist_ok=True)
                    saved_files[set_type][class_name] = []
            
            try:
                if hasattr(file, 'filename'):
                    safe_filename = sanitize_filename(file.filename)
                    file_path = os.path.join(class_folder, safe_filename)
                    
                    if hasattr(file, 'read') and callable(file.read):
                        contents = await file.read()
                        
                        with open(file_path, "wb") as f:
                            f.write(contents)
                        
                        file_size = os.path.getsize(file_path)
                        
                        if set_type in ["training_set", "test_set"]:
                            saved_files[set_type][class_name].append({
                                "original_filename": file.filename,
                                "saved_filename": safe_filename,
                                "path": file_path,
                                "size": file_size
                            })
                
            except Exception as e:
                print(f"Error saving file: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                return {"error": f"Failed to save file: {str(e)}"}
        
        training_results = train_and_evaluate_cnn(
            base_folder='./base', 
            epochs=epochs, 
            model_name=model_name, 
            layers=layers, 
            neurons_by_layer=neurons_by_layer
        )
        
        try:
            shutil.rmtree(parent_folder)
        except Exception as e:
            print(f"Error deleting folder: {str(e)}")
        
        return {
            "message": "Files uploaded and model trained successfully",
            "training_results": training_results,
        }
    
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}
    
@app.post("/cnn/classify-image/")
async def classify_image(
    image: UploadFile = File(...),
    model_name: str = Form(...)
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            contents = await image.read()
            temp_image.write(contents)
            temp_image_path = temp_image.name
        
        try:
            result = classify_new_image(model_name, temp_image_path)
            return {
                "model_name": model_name,
                "classification_result": result
            }
            
        finally:
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Classification error: {str(e)}\n{error_details}")
        return {"error": f"Classification failed: {str(e)}"}

@app.get("/cnn/list-models/")
async def list_models():
    try:
        models = list_trained_models()
        return {
            "models_count": len(models),
            "models": models
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error listing models: {str(e)}\n{error_details}")
        return {"error": f"Failed to list models: {str(e)}"}