from pydantic import BaseModel
from .colorRange import ColorRange
from typing import List


class ClassParameters(BaseModel):
    class_name: str
    rgb_ranges: List[ColorRange]
