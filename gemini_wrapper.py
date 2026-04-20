from google import genai
import os
from PIL import Image
from dotenv import load_dotenv

class GeminiOCR:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        # Initialize the modern Gemini client
        self.client = genai.Client(api_key=api_key)
        # Using gemini-3-flash-preview for high accuracy and the latest features
        self.model_id = 'gemini-3-flash-preview'

    def predict(self, image_pil):
        """
        Extract Hebrew text from an image using Gemini 3 Flash.
        image_pil: PIL.Image object
        """
        prompt = (
            "Extract all the Hebrew text from this image. "
            "The image is a student's handwritten test paper. "
            "Please preserve the reading order and layout as much as possible. "
            "Return only the extracted text."
        )
        
        try:
            # The new SDK accepts PIL images directly in the contents list
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt, image_pil]
            )
            return response.text
        except Exception as e:
            return f"[Gemini Error: {e}]"

if __name__ == "__main__":
    # Quick test
    try:
        ocr = GeminiOCR()
        print("GeminiOCR initialized successfully.")
    except Exception as e:
        print(f"Initialization failed: {e}")
