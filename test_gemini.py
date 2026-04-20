import os
from dotenv import load_dotenv
from gemini_wrapper import GeminiOCR
from PIL import Image

def test_gemini_connection():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("FAIL: GEMINI_API_KEY NOT FOUND IN .ENV")
        return

    print(f"API Key found (Length: {len(api_key)})")
    
    try:
        ocr = GeminiOCR()
        print(f"GeminiOCR initialized with model: {ocr.model_id}")
        
        print("Sending test request to Gemini...")
        # Verify API connectivity with a simple text prompt
        response = ocr.client.models.generate_content(
            model=ocr.model_id,
            contents="Say 'Gemini OK' if you are working."
        )
        
        if "Gemini OK" in response.text:
            print("SUCCESS: Gemini API is responsive.")
            print(f"Model Response: {response.text}")
        else:
            print(f"UNEXPECTED RESPONSE: {response.text}")
            
    except Exception as e:
        print(f"FAIL: Error during API test: {e}")

if __name__ == "__main__":
    test_gemini_connection()
