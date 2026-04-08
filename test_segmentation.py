import cv2
import numpy as np
from HebHTR.segmentation import segment_into_words

def create_dummy_image():
    # Create a white image
    img = np.ones((400, 600), dtype=np.uint8) * 255
    
    # Draw some "words" (black rectangles)
    # Line 1
    cv2.rectangle(img, (50, 50), (150, 100), 0, -1) # Word 1
    cv2.rectangle(img, (200, 50), (300, 100), 0, -1) # Word 2
    cv2.rectangle(img, (350, 50), (450, 100), 0, -1) # Word 3
    
    # Line 2
    cv2.rectangle(img, (100, 200), (200, 250), 0, -1) # Word 4
    cv2.rectangle(img, (250, 200), (350, 250), 0, -1) # Word 5
    
    return img

def test_segmentation():
    print("Creating dummy image with 5 words (black rectangles)...")
    img = create_dummy_image()
    
    print("Running segmentation...")
    results = segment_into_words(img)
    
    count = len(results)
    print(f"Detected {count} words.")
    
    if count == 5:
        print("PASS: Correct number of words detected.")
    else:
        print(f"FAIL: Expected 5 words, detected {count}.")

    for i, (crop, box) in enumerate(results):
        print(f"Word {i}: Box {box}")

if __name__ == "__main__":
    test_segmentation()
