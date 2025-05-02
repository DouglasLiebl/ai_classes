from fastapi import BackgroundTasks
from service.batch_upload_manager import BatchUploadManager
from entities.trainingParameters import TrainingParameters
from service.cnn import train_and_evaluate_cnn
from service.feature_extraction import train_neural_network_from_csv, extract_features
import os
import shutil
from datetime import datetime
import traceback


def background_model_training(
    session_id: str,
    params: TrainingParameters,
):
    try:
        BatchUploadManager.update_metadata(
            session_id, {"status": "training_in_progress"}
        )

        total_rgb_ranges = 0
        for training_class in params.rgb_ranges:
            total_rgb_ranges += len(training_class.rgb_ranges)

        parent_folder = f"base/{params.model_name}"
        training_folder = os.path.join(parent_folder, "training_set")

        if params.rgb_ranges is not None:
            features_csv = os.path.join(
                parent_folder, params.model_name + "_features.csv"
            )
            extract_features(
                main_folder=training_folder,
                output_csv=features_csv,
                classes=params.rgb_ranges,
            )

            training_results = train_neural_network_from_csv(
                features_csv,
                epochs=params.epochs,
                model_name=params.model_name,
                layers=params.layers,
                neurons_by_layer=params.neurons_by_layer,
                total_rgb_ranges=total_rgb_ranges
            )
        else:
            training_results = train_and_evaluate_cnn(
                base_folder=None,
                epochs=params.epochs,
                model_name=params.model_name or f"model_{session_id[:8]}",
                layers=params.layers,
                neurons_by_layer=params.neurons_by_layer,
            )

        BatchUploadManager.update_metadata(
            session_id,
            {
                "status": "completed",
                "total_rgb_ranges": total_rgb_ranges,
                "training_results": training_results,
                "completed_at": datetime.now().isoformat(),
            },
        )

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Background training error: {str(e)}\n{error_details}")
        BatchUploadManager.update_metadata(
            session_id,
            {
                "status": "error",
                "error": str(e),
                "failed_at": datetime.now().isoformat(),
            },
        )
