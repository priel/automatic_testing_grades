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
        Predict Hebrew text from a PIL Image (Single Word).
        Args:
            image_pil: PIL.Image object
        Returns:
            str: Decoded text
        """
        img_np = np.array(image_pil)
        
        # Check if image is RGB or RGBA, convert to grayscale
        if len(img_np.shape) == 3:
             img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
             img_gray = img_np

        # Preprocess
        target_size = (128, 32)
        processed_img = preprocessImageForPrediction(img_gray, target_size)
        
        # Batch wrapper as expected by Model.inferBatch
        class Batch:
            def __init__(self, imgs):
                self.imgs = np.stack(imgs, axis=0)
                self.gtTexts = None

        batch = Batch([processed_img])
        
        model = HebrewOCR.get_model()
        recognized = model.inferBatch(batch, True)[0] # Returns tuple (texts, probs)
        
        return recognized[0]

    @staticmethod
    def predict_full_page(image_pil, return_annotated=False, run_ocr=True):
        """
        Predict Hebrew text from a full-page PIL Image by segmenting it into words.
        Args:
            image_pil: PIL.Image object
            return_annotated: bool, if True returns (text, annotated_pil_image, boxes_data)
            run_ocr: bool, if False, skips text recognition and just returns the annotated image
        Returns:
            str: Decoded text (space-separated words) OR tuple IF return_annotated is True
        """
        # Lazy import
        try:
            from HebHTR.segmentation import segment_into_words
        except ImportError:
            import segmentation
            segment_into_words = segmentation.segment_into_words
        
        # Ensure image is RGB to avoid Palette-mode color flipping
        image_pil = image_pil.convert('RGB')
        img_np = np.array(image_pil)
        
        try:
            segment_results = segment_into_words(img_np)
        except Exception as e:
            print(f"Segmentation error: {e}")
            return f"[Segmentation Failed: {e}]"
            
        if not segment_results:
            return "[No text found]"
        
        annotated_img_pil = None
        boxes_text = ""
        if return_annotated:
            # Group words back into lines for visualization to color code them
            lines_of_boxes = []
            if segment_results:
                current_line_boxes = [segment_results[0][1]]
                for i in range(1, len(segment_results)):
                    box = segment_results[i][1] # (x,y,w,h)
                    prev_box = current_line_boxes[-1]
                    
                    # If Y diff is small, same line
                    if abs(box[1] - prev_box[1]) < 20: 
                        current_line_boxes.append(box)
                    else:
                        lines_of_boxes.append(current_line_boxes)
                        current_line_boxes = [box]
                lines_of_boxes.append(current_line_boxes)

            # Draw
            # img_np is RGB from PIL.Image
            img_draw = img_np.copy()
            if len(img_draw.shape) == 2:
                img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2RGB)
            elif len(img_draw.shape) == 3 and img_draw.shape[2] == 4:
                img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGBA2RGB)

            # RGB Colors for different lines
            colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), 
                      (0,255,255), (128,0,255), (255,128,0), (0,128,128), (128,128,0)]

            boxes_data = []
            for line_idx, line_boxes in enumerate(lines_of_boxes):
                color = colors[line_idx % len(colors)]
                for word_idx, (lx, ly, lw, lh) in enumerate(line_boxes):
                    cv2.rectangle(img_draw, (lx, ly), (lx + lw, ly + lh), color, 2)
                    boxes_data.append({
                        "line": line_idx + 1,
                        "word": word_idx + 1,
                        "box": (lx, ly, lw, lh)
                    })
            
            import PIL.Image
            annotated_img_pil = PIL.Image.fromarray(img_draw)
        # ---------------------------

        if not run_ocr:
            final_text = "[OCR Skipped - Displaying Segmentation Only]"
            if return_annotated:
                return final_text, annotated_img_pil, boxes_data
            return final_text

        img_gray_crops = [x[0] for x in segment_results]
        
        # Preprocess all crops
        target_size = (128, 32)
        processed_imgs = [preprocessImageForPrediction(crop, target_size) for crop in img_gray_crops]
        
        # Batch wrapper
        class Batch:
            def __init__(self, imgs):
                self.imgs = np.stack(imgs, axis=0)
                self.gtTexts = None
                
        batch = Batch(processed_imgs)
        
        model = HebrewOCR.get_model()
        recognized_texts = model.inferBatch(batch, True)[0] 
        
        final_text = " ".join(recognized_texts)
        
        if return_annotated:
            return final_text, annotated_img_pil, boxes_data
        return final_text
