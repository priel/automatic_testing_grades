import sys
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

# Add HebHTR directory to path so we can import its modules
HEBHTR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HebHTR')
if HEBHTR_DIR not in sys.path:
    sys.path.append(HEBHTR_DIR)

from Model import Model, DecoderType
from processFunctions import preprocessImageForPrediction

class HebrewOCR:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # Path to charList
            char_list_path = os.path.join(HEBHTR_DIR, 'model', 'charList.txt')
            if not os.path.exists(char_list_path):
                raise FileNotFoundError(f"Could not find charList.txt at {char_list_path}")
            
            with open(char_list_path, 'r', encoding='utf-8') as f:
                char_list = f.read()

            # Initialize model with BestPath decoder (SimpleHTR default, works on Windows)
            # mustRestore=True to load weights
            cls._model = Model(char_list, decoderType=DecoderType.BestPath, mustRestore=True)
        return cls._model

    @staticmethod
    def predict(image_pil):
        """
        Predict Hebrew text from a PIL Image.
        Args:
            image_pil: PIL.Image object
        Returns:
            str: Decoded text
        """
        # Convert PIL to cv2/numpy (Grayscale)
        # PIL RGB -> OpenCV BGR -> Grayscale
        # Or just PIL L -> numpy
        
        img_np = np.array(image_pil)
        
        # Check if image is RGB or RGBA, convert to grayscale
        if len(img_np.shape) == 3:
             img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
             img_gray = img_np

        # Preprocess
        # Model.imgSize is (128, 32) defined in Model class
        # We need to access it from the instance or class
        # accessing via Model class directly
        target_size = (128, 32)
        
        processed_img = preprocessImageForPrediction(img_gray, target_size)
        
        # Batch wrapper as expected by Model.inferBatch
        # HebHTR's Model.inferBatch expects an object with an 'imgs' attribute
        class Batch:
            def __init__(self, imgs):
                self.imgs = np.stack(imgs, axis=0)
                self.gtTexts = None

        batch = Batch([processed_img])
        
        model = HebrewOCR.get_model()
        recognized = model.inferBatch(batch, True)[0] # Returns tuple (texts, probs)
        
        # recognized is a list of strings (one per batch item)
        return recognized[0]
