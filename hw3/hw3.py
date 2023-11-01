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

        self.red_button = tk.Button(root, text="Show Red", command=self.red_part)
        self.red_button.grid(row=1, column=1, pady=5)

        self.green_button = tk.Button(root, text="Show Green", command=self.green_part)
        self.green_button.grid(row=2, column=1, pady=5)

        self.blue_button = tk.Button(root, text="Show Blue", command=self.blue_part)
        self.blue_button.grid(row=3, column=1, pady=5)

        self.hue_button = tk.Button(root, text="Show Hue", command=self.hue_part)
        self.hue_button.grid(row=1, column=2, pady=5)

        self.sat_button = tk.Button(root, text="Show Saturation", command=self.sat_part)
        self.sat_button.grid(row=2, column=2, pady=5)

        self.gray_button = tk.Button(root, text="Show Intensity", command=self.gray_part)
        self.gray_button.grid(row=3, column=2, pady=5)
        # Initializing
        self.original_image = None
        self.image = None

    def rgb2hsv(self):
        self.image = self.image.convert("HSV")
        self.image = self
        self.display_image()

    def hue_part(self):
        self.rgb2hsv()
        pass
    def sat_part(self):
        self.rgb2hsv()
        pass
    def gray_part(self):
        self.rgb2hsv()
        pass

    def rgb_part(self, color_id):
        # Splitting r,g,b image
        red, green, blue = self.image.split()
        # get a All 0s image
        temp, _, _ = Image.new("RGB", (512,512)).split()

        if color_id == 0:
            new_image = (red, temp, temp)
        elif color_id == 1:
            new_image = (temp, green, temp)
        elif color_id == 2:
            new_image = (temp, temp, blue)

        self.image = Image.merge("RGB", new_image)
        self.display_image()


    def red_part(self):
        self.rgb_part(0)
    def green_part(self):
        self.rgb_part(1)
    def blue_part(self):
        self.rgb_part(2)

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