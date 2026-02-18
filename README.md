# Multi Modal Document Understanding (MMDU)

he goal is to extract and understand information from documents containing a mix of images, text, and tables, associating them contextually. This approach moves beyond traditional OCR and basic LLM processing by leveraging specialized agents and models for each modality, orchestrated to achieve a holistic understanding of the document's content and structure.
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


