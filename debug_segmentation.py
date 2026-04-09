"""
Debug the two-pass segmentation pipeline.
Saves images for: lines detection, then words within each line.

Usage: python debug_segmentation.py <image_path>
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from HebHTR.segmentation import segment_into_lines, segment_line_into_words


def debug_pipeline(image_path):
    img_data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    out_dir = os.path.join(os.path.dirname(image_path), "debug")
    os.makedirs(out_dir, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imwrite(os.path.join(out_dir, "step0_threshold.png"), thresh)
    print("Saved step0_threshold.png")

    # === PASS 1: Lines ===
    lines = segment_into_lines(gray, thresh)
    print(f"\nPass 1: Found {len(lines)} lines")

    lines_img = img.copy()
    colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), 
              (255,255,0), (128,0,255), (0,128,255), (128,128,0), (0,128,128)]

    for i, (line_gray, line_thresh, (lx, ly, lw, lh)) in enumerate(lines):
        color = colors[i % len(colors)]
        cv2.rectangle(lines_img, (lx, ly), (lx+lw, ly+lh), color, 2)
        cv2.putText(lines_img, f"L{i+1}", (lx-30, ly + lh//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"  Line {i+1}: pos=({lx},{ly}) size={lw}x{lh}")

    cv2.imwrite(os.path.join(out_dir, "step1_lines.png"), lines_img)
    print("Saved step1_lines.png")

    # === PASS 2: Words per line ===
    words_img = img.copy()
    total_words = 0

    for i, (line_gray, line_thresh, (lx, ly, lw, lh)) in enumerate(lines):
        color = colors[i % len(colors)]
        words = segment_line_into_words(line_gray, line_thresh)
        print(f"\n  Line {i+1}: {len(words)} words")

        # Save individual line crop with word boxes
        line_debug = cv2.cvtColor(line_gray, cv2.COLOR_GRAY2BGR)
        for j, (word_crop, (wx, wy, ww, wh)) in enumerate(words):
            # Draw on line crop
            cv2.rectangle(line_debug, (wx, wy), (wx+ww, wy+wh), (0, 0, 255), 1)
            cv2.putText(line_debug, str(j), (wx, wy-2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

            # Draw on full image (global coords)
            gx, gy = lx + wx, ly + wy
            cv2.rectangle(words_img, (gx, gy), (gx+ww, gy+wh), color, 1)
            total_words += 1

        cv2.imwrite(os.path.join(out_dir, f"step2_line{i+1}_words.png"), line_debug)

    cv2.imwrite(os.path.join(out_dir, "step2_all_words.png"), words_img)
    print(f"\nTotal words detected: {total_words}")
    print(f"\nAll debug images saved to: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_segmentation.py <image_path>")
        sys.exit(1)
    debug_pipeline(sys.argv[1])
