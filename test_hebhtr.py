import os
import sys
from PIL import Image

try:
    from hebhtr_wrapper import HebrewOCR
    print("HebrewOCR imported successfully.")
except ImportError as e:
    print(f"Failed to import HebrewOCR: {e}")
    # Don't exit yet, maybe running before pip finish, but we want to see error
    

def test():
    # Create white image 128x32
    img = Image.new('RGB', (128, 32), color='white')
    print("Running initial prediction on blank image...")
    try:
        text = HebrewOCR.predict(img)
        print(f"Prediction result: '{text}'")
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if "HebrewOCR" in locals():
        test()
    else:
        print("Skipping test due to import error.")
