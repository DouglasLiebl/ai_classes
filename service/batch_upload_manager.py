import json
import uuid
import os
from datetime import datetime
import unicodedata
from typing import List, Dict
import shutil


def sanitize_filename(filename: str) -> str:
    filename = (
        unicodedata.normalize("NFKD", filename)
        .encode("ASCII", "ignore")
        .decode("ASCII")
    )
    filename = filename.replace(" ", "_")
    filename = str(uuid.uuid4()) + "_" + filename
    return filename


TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


class BatchUploadManager:
    @staticmethod
    def create_session() -> str:
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(TEMP_UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        metadata = {
            "created_at": datetime.now().isoformat(),
            "classes": {},
            "status": "created",
            "total_files": 0,
        }
        with open(os.path.join(session_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        return session_id

    @staticmethod
    def get_session_path(session_id: str) -> str:
        return os.path.join(TEMP_UPLOAD_DIR, session_id)

    @staticmethod
    def get_metadata(session_id: str) -> Dict:
        metadata_path = os.path.join(TEMP_UPLOAD_DIR, session_id, "metadata.json")
        if not os.path.exists(metadata_path):
            raise ValueError(f"Session {session_id} does not exist")

        with open(metadata_path, "r") as f:
            return json.load(f)

    @classmethod
    def update_metadata(cls, session_id, updates):
        session_path = cls.get_session_path(session_id)
        metadata_path = os.path.join(session_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            raise ValueError(f"Metadata not found for session {session_id}")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        metadata.update(updates)
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        return metadata

    @staticmethod
    def add_files(session_id: str, class_name: str, files: List[Dict]) -> Dict:
        session_dir = BatchUploadManager.get_session_path(session_id)
        class_dir = os.path.join(session_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        metadata = BatchUploadManager.get_metadata(session_id)

        if class_name not in metadata["classes"]:
            metadata["classes"][class_name] = {"count": 0, "files": []}

        for file_info in files:
            file_path = os.path.join(
                class_dir, sanitize_filename(file_info["filename"])
            )

            with open(file_path, "wb") as f:
                f.write(file_info["content"])

            metadata["classes"][class_name]["count"] += 1
            metadata["classes"][class_name]["files"].append(
                {
                    "original_filename": file_info["filename"],
                    "saved_path": file_path,
                    "size": len(file_info["content"]),
                }
            )

            metadata["total_files"] += 1

        with open(os.path.join(session_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        return metadata

    @staticmethod
    def list_sessions() -> List[Dict]:
        sessions = []

        for session_id in os.listdir(TEMP_UPLOAD_DIR):
            if os.path.isdir(os.path.join(TEMP_UPLOAD_DIR, session_id)):
                try:
                    metadata = BatchUploadManager.get_metadata(session_id)
                    sessions.append(
                        {
                            "session_id": session_id,
                            "created_at": metadata.get("created_at"),
                            "status": metadata.get("status"),
                            "total_files": metadata.get("total_files"),
                            "total_rgb_ranges": metadata.get("total_rgb_ranges"), 
                            "training_results": metadata.get("training_results"),
                            "classes": list(metadata.get("classes", {}).keys()),
                        }
                    )
                except Exception as e:
                    print(f"Error reading session {session_id}: {str(e)}")

        return sessions

    @staticmethod
    def clean_session(session_id: str, preserve_metadata: bool = True) -> bool:

        try:
            session_dir = BatchUploadManager.get_session_path(session_id)

            if not os.path.exists(session_dir):
                print(f"Session directory {session_id} not found")
                return False

            if preserve_metadata:
                try:
                    metadata = BatchUploadManager.get_metadata(session_id)
                    metadata["status"] = "cleaned"
                    metadata["cleaned_at"] = datetime.now().isoformat()

                    class_summary = {}
                    for class_name, info in metadata["classes"].items():
                        class_summary[class_name] = info["count"]

                    metadata["class_summary"] = class_summary

                    for class_name in metadata["classes"]:
                        metadata["classes"][class_name]["files"] = []

                    metadata_path = os.path.join(session_dir, "metadata.json")
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f)
                except Exception as e:
                    print(f"Error updating metadata before cleaning: {str(e)}")

                for item in os.listdir(session_dir):
                    item_path = os.path.join(session_dir, item)
                    if item != "metadata.json":
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
            else:
                shutil.rmtree(session_dir)

            print(f"Session {session_id} cleaned successfully")
            return True

        except Exception as e:
            print(f"Error cleaning session {session_id}: {str(e)}")
            return False

    @staticmethod
    def upload_single_image(image_data: bytes, filename: str, class_name: str = "default") -> str:
        session_id = BatchUploadManager.create_session()
        
        file_info = {
            "filename": filename,
            "content": image_data
        }
        
        BatchUploadManager.add_files(session_id, class_name, [file_info])
        
        BatchUploadManager.update_metadata(session_id, {
            "is_single_image": True,
            "original_filename": filename,
            "class_name": class_name
        })
        
        return session_id
    
    @staticmethod
    def delete_session(session_id: str) -> bool:
        try:
            session_dir = BatchUploadManager.get_session_path(session_id)
            
            if not os.path.exists(session_dir):
                print(f"Session directory {session_id} not found")
                return False
                
            shutil.rmtree(session_dir)
            print(f"Session {session_id} deleted successfully")
            return True
            
        except Exception as e:
            print(f"Error deleting session {session_id}: {str(e)}")
            return False