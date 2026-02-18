from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class OCRModel(ABC):
    @abstractmethod
    def extract_text(self, image: Any) -> List[Dict[str, Any]]:
        """
        Extract text and bounding boxes from an image.
        Returns a list of dicts: [{"text": str, "bbox": [l, t, r, b]}]
        """
        pass

class LayoutModel(ABC):
    @abstractmethod
    def predict_order(self, boxes: List[List[int]]) -> List[int]:
        """
        Predict the reading order of the given bounding boxes.
        Returns a list of indices representing the order.
        """
        pass

class VLMModel(ABC):
    @abstractmethod
    def analyze(self, image: Any, prompt: str) -> str:
        """
        Analyze an image or a region of an image using a VLM.
        """
        pass
