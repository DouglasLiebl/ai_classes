from fastapi import FastAPI
import os
from datetime import datetime
from fastapi import Request
import unicodedata
import shutil
from cnn import train_and_evaluate_cnn

app = FastAPI()

def sanitize_filename(filename: str) -> str:
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    filename = filename.replace(' ', '_')
    return filename

@app.post("/upload-images")
async def upload_images(request: Request, epochs: int = 30):
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
        print(form_data.multi_items())

        for field_name, file in form_data.multi_items():
            print(f"Processing file: {file.filename} from field: {field_name}")
            
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
                    print(f"Saving to: {file_path}")
                    
                    if hasattr(file, 'read') and callable(file.read):
                        contents = await file.read()
                        print(f"Read {len(contents)} bytes")
                        
                        with open(file_path, "wb") as f:
                            f.write(contents)
                        
                        file_size = os.path.getsize(file_path)
                        print(f"Saved file with size: {file_size}")
                        
                        if set_type in ["training_set", "test_set"]:
                            saved_files[set_type][class_name].append({
                                "original_filename": file.filename,
                                "saved_filename": safe_filename,
                                "path": file_path,
                                "size": file_size
                            })
                    else:
                        print(f"File object doesn't have a read method")
                else:
                    print(f"Object is not a file or doesn't have filename attribute")
                
            except Exception as e:
                print(f"Error saving file: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                return {"error": f"Failed to save file: {str(e)}"}
        
        print("Starting model training...")
        training_results = train_and_evaluate_cnn(base_folder='./base', epochs=epochs)
        print(f"Model training completed with accuracy: {training_results['accuracy']}")
        
        try:
            shutil.rmtree(parent_folder)
            print(f"Base folder {parent_folder} deleted successfully")
        except Exception as e:
            print(f"Error deleting folder: {str(e)}")
        
        return {
            "message": "Files uploaded and model trained successfully",
            "training_results": training_results,
            "folders_structure": {
                "training_set": list(saved_files["training_set"].keys()),
                "test_set": list(saved_files["test_set"].keys())
            }
        }
    except Exception as e:
        print(f"Global error: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}