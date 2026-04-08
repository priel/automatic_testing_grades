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
    def predict_full_page(image_pil, debug=False):
        """
        Predict Hebrew text from a full-page PIL Image by segmenting it into words.
        Args:
            image_pil: PIL.Image object
            debug: bool, if True show popup with boxes
        Returns:
            str: Decoded text (space-separated words)
        """
        # Lazy import
        try:
            from HebHTR.segmentation import segment_into_words
        except ImportError:
            import segmentation
            segment_into_words = segmentation.segment_into_words
        
        img_np = np.array(image_pil)
        
        try:
            segment_results = segment_into_words(img_np)
        except Exception as e:
            print(f"Segmentation error: {e}")
            return f"[Segmentation Failed: {e}]"
            
        if not segment_results:
            return "[No text found]"
        
        # --- DEBUG VISUALIZATION (Lines) ---
        if debug:
            # Group words back into lines for visualization
            # We know they are sorted. We can group by Y proximity.
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

            # Show each line
            print(f"Debug: Detected {len(lines_of_boxes)} lines.")
            img_bgr = img_np.copy()
            if len(img_bgr.shape) == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

            for line_idx, line_boxes in enumerate(lines_of_boxes):
                # Calculate bounding box of the whole line
                min_x = min(b[0] for b in line_boxes)
                min_y = min(b[1] for b in line_boxes)
                max_x = max(b[0] + b[2] for b in line_boxes)
                max_y = max(b[1] + b[3] for b in line_boxes)
                
                # Crop and show loop
                line_img = img_bgr[min_y:max_y, min_x:max_x].copy()
                
                # Draw boxes for words in this line
                for (lx, ly, lw, lh) in line_boxes:
                    # Adjust coordinates to be relative to the line crop
                    cv2.rectangle(line_img, (lx - min_x, ly - min_y), (lx - min_x + lw, ly - min_y + lh), (0, 0, 255), 1)

                title = f"Debug: Line {line_idx+1}/{len(lines_of_boxes)}"
                cv2.imshow(title, line_img)
                print(f"Showing Line {line_idx+1}. Press 'q' to quit debug, any other key to continue...")
                
                key = cv2.waitKey(0) # Wait for key
                
                try:
                    cv2.destroyWindow(title)
                except Exception:
                    # Window likely closed by user
                    pass
                
                if key & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
        # ---------------------------

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
        return final_text
