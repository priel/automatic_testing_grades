import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import pytesseract
import os

# NOTE: You must have Tesseract-OCR installed on your system.
# Windows installer: https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class AutoGraderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Grader - Local Version")
        self.root.geometry("1000x800")

        # variables
        self.current_image = None
        self.photo_image = None

        self.setup_ui()

    def setup_ui(self):
        # 1. Top Bar: Buttons only (No API Key needed)
        top_frame = tk.Frame(self.root, pady=10, padx=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        btn_load = tk.Button(top_frame, text="Select Image", command=self.load_image)
        btn_load.pack(side=tk.LEFT, padx=10)

        self.btn_grade = tk.Button(top_frame, text="Scan (OCR)", command=self.process_image, state=tk.DISABLED, bg="#dddddd")
        self.btn_grade.pack(side=tk.LEFT, padx=10)

        # Hebrew HTR Toggle
        self.use_heb_htr = tk.BooleanVar()
        self.chk_heb = tk.Checkbutton(top_frame, text="Hebrew Handmade (HebHTR - Word)", variable=self.use_heb_htr)
        self.chk_heb.pack(side=tk.LEFT, padx=10)

        # 2. Main Content Area (Split: Left=Image, Right=Text)
        content_frame = tk.Frame(self.root)
        content_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Left: Image
        self.image_label = tk.Label(content_frame, text="No image selected", bg="gray", width=50)
        self.image_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))

        # Right: Output Text
        self.text_output = scrolledtext.ScrolledText(content_frame, width=40, font=("Consolas", 10))
        self.text_output.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.text_output.insert(tk.END, "Status: Ready to load image...\n")
        self.log("Local OCR Mode (Tesseract)")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Test Paper",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")]
        )
        
        if file_path:
            try:
                self.current_image = Image.open(file_path)
                
                # Display logic
                display_h = 600
                ratio = display_h / self.current_image.height
                display_w = int(self.current_image.width * ratio)
                
                resized = self.current_image.copy().resize((display_w, display_h))
                self.photo_image = ImageTk.PhotoImage(resized)
                
                self.image_label.config(image=self.photo_image, text="")
                self.btn_grade.config(state=tk.NORMAL, bg="#4CAF50", fg="white")
                self.log("Image loaded successfully.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def log(self, message):
        self.text_output.insert(tk.END, f"\n{message}")
        self.text_output.see(tk.END)

    def process_image(self):
        if not self.current_image:
            return

        if self.use_heb_htr.get():
            self.process_image_hebhtr()
            return

        self.log("Starting Tesseract OCR...")
        self.root.update()

        try:
            # Check available languages
            # Note: We are hardcoding the path for now based on previous steps
            tess_cmd = pytesseract.pytesseract.tesseract_cmd
            tess_dir = os.path.dirname(tess_cmd)
            
            # Helper to run OCR
            # We attempt to use Hebrew and English. 
            # If Hebrew data is missing, Tesseract might error or default back.
            text = pytesseract.image_to_string(self.current_image, lang='heb+eng')
            
            self.log("--- OCR RESULT (Hebrew/English) ---")
            if not text.strip():
                self.log("[No text detected. Try a clearer image.]")
            else:
                self.log(text)
            self.log("------------------")

        except pytesseract.TesseractError as e:
            if "tessdata" in str(e) or "heb" in str(e):
                self.log("ERROR: Hebrew language data not found!")
                self.log("Please download 'heb.traineddata' and place it in the 'tessdata' folder.")
                messagebox.showerror("Missing Hebrew Data", "To scan Hebrew, you must download the language file.\nSee instructions in the log.")
            else:
                 self.log(f"Tesseract Error: {e}")
                 messagebox.showerror("OCR Error", f"Tesseract Error:\n{e}")


        except pytesseract.TesseractNotFoundError:
            self.log("ERROR: Tesseract is not found in your PATH.")
            self.log("Please install Tesseract-OCR and add it to PATH.")
            self.log("Download: https://github.com/UB-Mannheim/tesseract/wiki")
            messagebox.showerror("Tesseract Missing", "Tesseract executable not found.\nPlease install Tesseract-OCR.")
            self.log(f"ERROR: {e}")
            messagebox.showerror("OCR Error", f"Failed to process image:\n{e}")

    def process_image_hebhtr(self):
        self.log("Starting HebHTR (Hebrew Handwriting)...")
        self.log("NOTE: HebHTR expects a single word image.")
        self.root.update()
        
        try:
            # Lazy import to allow UI to load even if dependencies are missing
            from hebhtr_wrapper import HebrewOCR
            
            # Predict
            # Using predict_full_page to handle segmentation of words
            # Debug=True to show segmentation boxes
            text = HebrewOCR.predict_full_page(self.current_image, debug=True)
            
            self.log("--- HebHTR RESULT (Segmented) ---")
            self.log(f"{text}")
            self.log("---------------------------------")
            
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

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoGraderApp(root)
    root.mainloop()
