from pydantic import BaseModel
from typing import List


class ColorRange(BaseModel):
    name: str
    rgb: List[int]
