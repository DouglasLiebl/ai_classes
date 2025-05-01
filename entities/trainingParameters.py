from pydantic import BaseModel
from typing import List, Optional
from .classParameters import ClassParameters


class TrainingParameters(BaseModel):
    epochs: int = 30
    model_name: str = ""
    layers: int = 6
    neurons_by_layer: int = 6
    test_percentage: float = 20.0
    rgb_ranges: Optional[List[ClassParameters]] = None
