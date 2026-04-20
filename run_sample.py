import os
from gemini_wrapper import GeminiOCR
from PIL import Image
from dotenv import load_dotenv

def run_sample_ocr():
    load_dotenv()
    image_path = os.path.join("examples", "sample.png")
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    
    print("Initializing GeminiOCR...")
    ocr = GeminiOCR()
    
    print("Running OCR...")
    text = ocr.predict(image)
    
    print("\n--- OCR Result ---")
    print(text)
    print("------------------")

if __name__ == "__main__":
    run_sample_ocr()
