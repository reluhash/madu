import argparse
import os
import cv2
from factory import ModelFactory

def main(args):
    # Initialize models
    ocr_model = ModelFactory.get_ocr_model(args.ocr_provider)
    layout_model = ModelFactory.get_layout_model(args.layout_provider)
    vlm_model = ModelFactory.get_vlm_model(args.vlm_provider) if args.vlm_provider else None

    # Load image
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not load image: {args.image_path}")

    # Step 1: OCR
    print(f"Running OCR using {args.ocr_provider}...")
    ocr_results = ocr_model.extract_text(image)
    
    # Extract boxes and text
    boxes = [res["bbox"] for res in ocr_results]
    texts = [res["text"] for res in ocr_results]

    # Step 2: Layout/Reading Order
    print(f"Predicting reading order using {args.layout_provider}...")
    order_indices = layout_model.predict_order(boxes)

    # Reorder texts
    ordered_texts = [texts[idx] for idx in order_indices]
    
    print("\n--- Ordered Text ---")
    print(" ".join(ordered_texts))

    # Step 3: Optional VLM Analysis
    if vlm_model:
        print(f"\nRunning VLM analysis using {args.vlm_provider}...")
        vlm_result = vlm_model.analyze(image, "Summarize this document.")
        print(f"VLM Result: {vlm_result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular Document Understanding")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the document image")
    parser.add_argument("--ocr_provider", type=str, default="paddleocr", help="OCR provider name")
    parser.add_argument("--layout_provider", type=str, default="layoutlmv3", help="Layout provider name")
    parser.add_argument("--vlm_provider", type=str, default=None, help="VLM provider name")
    
    args = parser.parse_args()
    main(args)
