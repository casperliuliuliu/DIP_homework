import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFilter
from math import exp, log
import numpy as np

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")

        # Image field
        self.image_label = tk.Label(root)
        self.open_button = tk.Button(root, text="Open Image", command=self.open_image)
        self.save_button = tk.Button(root, text="Save Image", command=self.save_image)
        self.recover_button = tk.Button(root, text="Recover Image", command=self.recover_image)
        self.message_label = tk.Label(root, text="", fg="green")
        self.error_label = tk.Label(root, text="", fg="red")
        # column 0
        self.image_label.grid(row=0, column=0, columnspan=3, pady=10)
        self.open_button.grid(row=1, column=0, pady=5)
        self.save_button.grid(row=2, column=0, pady=5)
        self.recover_button.grid(row=3, column=0, pady=5)
        self.message_label.grid(row=11, column=0, columnspan=3)
        self.error_label.grid(row=12, column=0, columnspan=3)

        self.red_button = tk.Button(root, text="Red Part", command=self.red_part)

        # Initializing
        self.original_image = None
        self.image = None
    def rgb_part(self, color):
        pass
    
    def red_part(self):
        self.rgb_part('r')

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.tif *.tiff")])
        try:
            self.image = Image.open(file_path)
            self.original_image = self.image.copy() # Backup
            self.display_image()
        except Exception as e:
            self.error_label.config(text=f"Error: {str(e)}")

    def save_image(self):
        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg *.jpeg"), ("TIF files", "*.tif *.tiff")])
            self.image.save(file_path)
            messagebox.showinfo("Save Image", "Image saved successfully!")
        except Exception as e:
            self.error_label.config(text=f"Error: {str(e)}")

    def display_image(self):
        try:
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo
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