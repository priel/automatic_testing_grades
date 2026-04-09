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

        self.btn_scan = tk.Button(top_frame, text="Scan (HebHTR)", command=self.process_image, state=tk.DISABLED, bg="#dddddd")
        self.btn_scan.pack(side=tk.LEFT, padx=10)

        # Main Content Area (Split: Left=Image, Right=Text)
        content_frame = tk.Frame(self.root)
        content_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Left: Image
        self.image_label = tk.Label(content_frame, text="No image selected", bg="gray", width=50)
        self.image_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))

        # Right: Output Text
        self.text_output = scrolledtext.ScrolledText(content_frame, width=40, font=("Consolas", 10))
        self.text_output.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.text_output.insert(tk.END, "Status: Ready to load image...\n")
        self.log("Mode: Hebrew Handwriting Recognition (HebHTR)")

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
                self.btn_scan.config(state=tk.NORMAL, bg="#4CAF50", fg="white")
                self.log("Image loaded successfully.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def log(self, message):
        self.text_output.insert(tk.END, f"\n{message}")
        self.text_output.see(tk.END)

    def process_image(self):
        if not self.current_image:
            return

        self.log("Starting HebHTR (Hebrew Handwriting)...")
        self.root.update()
        
        try:
            from hebhtr_wrapper import HebrewOCR
            
            # Debug=True to show segmentation boxes
            text = HebrewOCR.predict_full_page(self.current_image, debug=True)
            
            self.log("--- HebHTR RESULT ---")
            self.log(f"{text}")
            self.log("---------------------")
            
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
