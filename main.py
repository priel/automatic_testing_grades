import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import os

class AutoGraderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Grader - Hebrew Handwriting")
        self.root.geometry("1000x800")

        # variables
        self.current_image = None
        self.photo_image = None

        self.setup_ui()

    def setup_ui(self):
        # Top Bar
        top_frame = tk.Frame(self.root, pady=10, padx=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        btn_load = tk.Button(top_frame, text="Select Image", command=self.load_image)
        btn_load.pack(side=tk.LEFT, padx=10)

        self.btn_segment = tk.Button(top_frame, text="View Segmentation", command=lambda: self.process_image(run_ocr=False), state=tk.DISABLED, bg="#dddddd")
        self.btn_segment.pack(side=tk.LEFT, padx=10)

        self.btn_scan = tk.Button(top_frame, text="Run Full OCR (Local)", command=lambda: self.process_image(run_ocr=True), state=tk.DISABLED, bg="#dddddd")
        self.btn_scan.pack(side=tk.LEFT, padx=10)

        self.btn_gemini = tk.Button(top_frame, text="Run Gemini OCR (Cloud)", command=self.run_gemini_ocr, state=tk.DISABLED, bg="#dddddd")
        self.btn_gemini.pack(side=tk.LEFT, padx=10)

        # Main Content Area (Split: Left=Image, Right=Text)
        content_frame = tk.Frame(self.root)
        content_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Left: Image
        self.image_label = tk.Label(content_frame, text="No image selected", bg="gray", width=50)
        self.image_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))

        # Right: Output Text
        self.text_output = scrolledtext.ScrolledText(content_frame, width=40, font=("Consolas", 12))
        self.text_output.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Configure right alignment tag for Hebrew
        self.text_output.tag_configure("right", justify='right')
        
        self.text_output.insert(tk.END, "Status: Ready to load image...\n", "right")
        self.log("Mode: Hebrew Handwriting Recognition (HebHTR)")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Test Paper",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.current_image = Image.open(file_path)
                
                # Display logic
                resized = self.current_image.copy()
                resized.thumbnail((550, 700))
                self.photo_image = ImageTk.PhotoImage(resized)
                
                self.image_label.config(image=self.photo_image, text="")
                self.btn_scan.config(state=tk.NORMAL, bg="#4CAF50", fg="white")
                self.btn_segment.config(state=tk.NORMAL, bg="#2196F3", fg="white")
                self.btn_gemini.config(state=tk.NORMAL, bg="#9C27B0", fg="white")
                self.log("Image loaded successfully.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def log(self, message):
        self.text_output.insert(tk.END, f"\n{message}", "right")
        self.text_output.see(tk.END)

    def process_image(self, run_ocr=True):
        if not self.current_image:
            return

        if run_ocr:
            self.log("Starting HebHTR (Hebrew Handwriting Full OCR)...")
        else:
            self.log("Starting Segmentation (Skipping OCR)...")
        self.root.update()
        
        try:
            from hebhtr_wrapper import HebrewOCR
            
            # return_annotated=True to get the image with segmentation boxes
            text, annotated_image, boxes_data = HebrewOCR.predict_full_page(self.current_image, return_annotated=True, run_ocr=run_ocr)
            
            # Format text for Tkinter display
            display_text = text
            if run_ocr and text and not text.startswith("["):
                try:
                    from bidi.algorithm import get_display
                    # Apply get_display to the full logical text to handle RTL ordering correctly
                    display_text = get_display(text)
                except ImportError:
                    pass
            
            self.log("--- HebHTR RESULT ---")
            self.log(f"{display_text}")
            self.log("---------------------")

            # Save boxes to output folder
            if hasattr(self, 'current_image_path') and self.current_image_path:
                img_dir = os.path.dirname(self.current_image_path)
                out_dir = os.path.join(img_dir, "output")
                os.makedirs(out_dir, exist_ok=True)
                
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                out_file = os.path.join(out_dir, f"{base_name}_coordinates.txt")
                
                boxes_text_lines = []
                crops_dir = os.path.join(out_dir, f"{base_name}_crops")
                os.makedirs(crops_dir, exist_ok=True)
                
                for b in boxes_data:
                    lx, ly, lw, lh = b['box']
                    L = b['line']
                    W = b['word']
                    boxes_text_lines.append(f"Line {L}, Word {W}: x={lx}, y={ly}, w={lw}, h={lh}")
                    
                    crop_box = (lx, ly, lx + lw, ly + lh)
                    word_crop = self.current_image.crop(crop_box)
                    crop_file = os.path.join(crops_dir, f"L{L}_W{W}.png")
                    word_crop.save(crop_file)
                
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(boxes_text_lines))
                
                out_text_file = os.path.join(out_dir, f"{base_name}_text.txt")
                with open(out_text_file, "w", encoding="utf-8") as f:
                    f.write(text)
                
                self.log(f"Saved coordinates to: {os.path.basename(out_file)}")
                self.log(f"Saved text output to: {os.path.basename(out_text_file)}")
                self.log(f"Saved {len(boxes_data)} crop images to: {os.path.basename(crops_dir)}")
            
            # Display annotated image
            if annotated_image:
                resized = annotated_image.copy()
                resized.thumbnail((550, 700))
                self.photo_image = ImageTk.PhotoImage(resized)
                self.image_label.config(image=self.photo_image, text="")
                
            
        except ImportError as e:
            self.log(f"ERROR: Import failed: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Import Error", f"Failed to import dependencies.\nError: {e}\n\nPlease ensure you are running in the correct environment.")
        except Exception as e:
            self.log(f"HebHTR Error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("HebHTR Error", f"Failed to run HebHTR:\n{e}")

    def run_gemini_ocr(self):
        if not self.current_image:
            return
        
        self.log("Starting Gemini OCR (Cloud - High Accuracy)...")
        self.root.update()
        
        try:
            from gemini_wrapper import GeminiOCR
            ocr = GeminiOCR()
            text = ocr.predict(self.current_image)
            
            # The user reported that characters are correct but word order is reversed.
            # We reverse only the word order for each line to fix the RTL layout in the UI.
            processed_lines = []
            for line in text.split("\n"):
                words = line.split(" ")
                processed_lines.append(" ".join(words[::-1]))
            display_text = "\n".join(processed_lines)
            
            self.log("--- Gemini RESULT ---")
            self.log(display_text)
            self.log("----------------------")
            
            # Save to output folder
            if hasattr(self, 'current_image_path') and self.current_image_path:
                img_dir = os.path.dirname(self.current_image_path)
                out_dir = os.path.join(img_dir, "output")
                os.makedirs(out_dir, exist_ok=True)
                
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                out_text_file = os.path.join(out_dir, f"{base_name}_gemini.txt")
                with open(out_text_file, "w", encoding="utf-8") as f:
                    f.write(text)
                self.log(f"Saved Gemini output to: {os.path.basename(out_text_file)}")
                
        except Exception as e:
            self.log(f"Gemini Error: {e}")
            messagebox.showerror("Gemini Error", f"Failed to run Gemini OCR:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoGraderApp(root)
    root.mainloop()
# Dummy change for git push test
