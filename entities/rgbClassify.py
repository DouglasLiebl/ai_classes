from pydantic import BaseModel
from .colorRange import ColorRange
from typing import List, Optional


class RgbClassify(BaseModel):
    model_name: str
    class_name: Optional[str] = None
    rgb_ranges: List[ColorRange]
