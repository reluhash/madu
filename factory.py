from providers import LayoutLMv3Provider, PaddleOCRProvider, HFVLMProvider
from modules import LayoutModel, OCRModel, VLMModel
from typing import Dict, Type

class ModelFactory:
    _layout_providers: Dict[str, Type[LayoutModel]] = {
        "layoutlmv3": LayoutLMv3Provider,
    }
    
    _ocr_providers: Dict[str, Type[OCRModel]] = {
        "paddleocr": PaddleOCRProvider,
    }
    
    _vlm_providers: Dict[str, Type[VLMModel]] = {
        "huggingface": HFVLMProvider,
    }

    @classmethod
    def get_layout_model(cls, name: str, **kwargs) -> LayoutModel:
        provider = cls._layout_providers.get(name.lower())
        if not provider:
            raise ValueError(f"Unknown layout provider: {name}")
        return provider(**kwargs)

    @classmethod
    def get_ocr_model(cls, name: str, **kwargs) -> OCRModel:
        provider = cls._ocr_providers.get(name.lower())
        if not provider:
            raise ValueError(f"Unknown OCR provider: {name}")
        return provider(**kwargs)

    @classmethod
    def get_vlm_model(cls, name: str, **kwargs) -> VLMModel:
        provider = cls._vlm_providers.get(name.lower())
        if not provider:
            raise ValueError(f"Unknown VLM provider: {name}")
        return provider(**kwargs)
