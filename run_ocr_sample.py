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
    
    # Save to a UTF-8 file to avoid console encoding issues
    output_file = "gemini_ocr_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"\nOCR Result saved to: {output_file}")
    
    # Also print a word-reversed version line-by-line to see if logic works
    print("\n--- Word-Reversed Output (Word-by-Word Reversal) ---")
    processed_lines = []
    for line in text.split("\n"):
        words = line.split(" ")
        processed_lines.append(" ".join(words[::-1]))
    
    # Save the word-reversed version to a file
    with open("gemini_ocr_word_reversed.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(processed_lines))
    print("Word-reversed output saved to: gemini_ocr_word_reversed.txt")

if __name__ == "__main__":
    run_sample_ocr()
