import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageFilter
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFilter
import numpy as np
import cv2

# Function to open and process images
class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")

        # Image field
        self.image_label = tk.Label(root)
        self.image_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        self.open_button = tk.Button(root, text="Open Image", command=self.open_image)
        self.open_button.grid(row=1, column=0, pady=5)
        self.save_button = tk.Button(root, text="Save Image", command=self.save_image)
        self.save_button.grid(row=2, column=0, pady=5)
        self.recover_button = tk.Button(root, text="Recover Image", command=self.recover_image)
        self.recover_button.grid(row=3, column=0, pady=5)
        self.message_label = tk.Label(root, text="", fg="green")
        self.message_label.grid(row=11, column=0, columnspan=3)
        self.error_label = tk.Label(root, text="", fg="red")
        self.error_label.grid(row=12, column=0, columnspan=3)
        
        self.average_button = tk.Button(root, text="Avg Filter", command=self.average_filter)
        self.average_button.grid(row=1, column=1, pady=5)

        self.median_button = tk.Button(root, text="Med Filter", command=self.median_filter)
        self.median_button.grid(row=2, column=1, pady=5)

        self.laplacian_button = tk.Button(root, text="Lap Filter", command=self.laplacian_filter)
        self.laplacian_button.grid(row=3, column=1, pady=5)
        self.label = tk.Label(root, text="Current State:")
        self.label.pack(pady=10)

        self.state_label = tk.Label(root, textvariable=current_state)
        self.state_label.pack()

        self.self.toggle_button = tk.Button(root, text="Toggle", command=toggle_state)
        self.toggle_button.pack(pady=10)
        current_state = tk.StringVar()
        current_state.set('A')
    def toggle_state():
        global current_state
        if current_state.get() == 'A':
            current_state.set('B')
        else:
            current_state.set('A')

root = tk.Tk()
root.title("Switch Example")
    def average_filter(self):
        kernel = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        kernel_size = (3, 3)
         # Apply the convolution with the custom kernel
        self.image = self.image.filter(ImageFilter.Kernel(kernel_size, kernel, scale=9, offset=0))
        self.display_image()

    def median_filter(self):
        self.image = self.image.filter(ImageFilter.MedianFilter(size=3))
        self.display_image()

    def laplacian_filter(self):
        kernel = [0, 1, 0, 1, -4, 1, 0, 1, 0]
        self.image = self.image.filter(ImageFilter.Kernel((3, 3), kernel))
        self.display_image()

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.tif *.tiff")])
        try:
            self.image = Image.open(file_path)
            self.image = self.image.convert("L")  # Convert to grayscale
            self.original_image = self.image.copy() # Backup
            self.display_image()
        except Exception as e:
            self.error_label.config(text=f"Error: {str(e)}")
    def display_image(self):
        try:
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo
        except Exception as e:
            self.error_label.config(text=f"Error: {str(e)}")

    def save_image(self):
        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg *.jpeg"), ("TIF files", "*.tif *.tiff")])
            self.image.save(file_path)
            messagebox.showinfo("Save Image", "Image saved successfully!")
        except Exception as e:
            self.error_label.config(text=f"Error: {str(e)}")
    
    def recover_image(self):
        if self.original_image:
            self.image = self.original_image.copy()
            self.display_image()
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()