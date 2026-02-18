# Modular Document Understanding (ADE)

This module refactors the original `ade` repository into a modular architecture, allowing for easy swapping of OCR, Layout, and Vision-Language Models (VLMs) while ensuring everything runs locally.

## Architecture Overview

The system is built on three core abstractions defined in `modules.py`:

1.  **OCRModel**: Extracts text and bounding boxes from document images.
2.  **LayoutModel**: Predicts the reading order of extracted bounding boxes.
3.  **VLMModel**: Performs complex analysis (e.g., summarization, chart analysis) on document regions.

## Key Files

-   `modules.py`: Abstract base classes for all model types.
-   `providers.py`: Implementations of specific models (e.g., `LayoutLMv3Provider`, `PaddleOCRProvider`).
-   `factory.py`: A central registry and factory to instantiate models by name.
-   `run_modular.py`: A CLI tool to run the entire pipeline with custom model combinations.
-   `setup_local.sh`: A shell script to set up the local environment and dependencies.

## Local Setup

To run everything locally, follow these steps:

1.  **Install Dependencies**:
    ```bash
    bash setup_local.sh
    ```

2.  **Run the Pipeline**:
    You can specify which providers to use for each step.
    ```bash
    python3 run_modular.py --image_path path/to/your/document.png \
                           --ocr_provider paddleocr \
                           --layout_provider layoutlmv3
    ```

## Adding New Models

To add a new model (e.g., a new OCR engine):

1.  Create a new class in `providers.py` that inherits from the appropriate base class in `modules.py`.
2.  Implement the required abstract methods.
3.  Register your new provider in `factory.py`.

## Supported Providers

| Task | Provider Name | Implementation |
| :--- | :--- | :--- |
| **OCR** | `paddleocr` | Local PaddleOCR engine |
| **Layout** | `layoutlmv3` | Local LayoutLMv3-based Reading Order model |
| **VLM** | `huggingface` | Placeholder for local HF-based VLMs |

## Benefits of Modularity

-   **Flexibility**: Easily test different OCR engines (e.g., Tesseract vs. PaddleOCR) or Layout models.
-   **Local Execution**: All providers are designed to run on local hardware (CPU/GPU).
-   **Extensibility**: Add support for proprietary APIs (like OpenAI or Anthropic) by simply adding a new provider.
-   **Clean Code**: Decouples the model logic from the application logic.
