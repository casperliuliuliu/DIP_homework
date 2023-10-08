import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from math import exp, log
import numpy as np
class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")

        self.image_label = tk.Label(root)
        self.image_label.grid(row=0, column=0, columnspan=3, pady=10)

        self.open_button = tk.Button(root, text="Open Image", command=self.open_image)
        self.open_button.grid(row=1, column=0, pady=5)

        self.save_button = tk.Button(root, text="Save Image", command=self.save_image)
        self.save_button.grid(row=2, column=0, pady=5)

        self.a_label = tk.Label(root, text="a/lower")
        self.a_label.grid(row=1, column=2)  # Adjust the padx value as needed
        self.a_entry = tk.Entry(root, width=10)
        self.a_entry.grid(row=2, column=2)
        
        self.b_label = tk.Label(root, text="b/upper")
        self.b_label.grid(row=3, column=2)  # Adjust the padx value as needed
        self.b_entry = tk.Entry(root, width=10)
        self.b_entry.grid(row=4, column=2)
        # self.b_entry.insert(0, "hehe")  # Insert the placeholder text

        self.linear_button = tk.Button(root, text="Linear", command=self.adjust_image_linear)
        self.linear_button.grid(row=1, column=1, pady=5)

        self.exponential_button = tk.Button(root, text="Exponential", command=self.adjust_image_exponential)
        self.exponential_button.grid(row=2, column=1, pady=5)

        self.logarithmic_button = tk.Button(root, text="Logarithmic", command=self.adjust_image_logarithmic)
        self.logarithmic_button.grid(row=3, column=1, pady=5)

        self.zoom_in_button = tk.Button(root, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.grid(row=4, column=0, pady=5)

        self.shrink_button = tk.Button(root, text="Shrink", command=self.shrink)
        self.shrink_button.grid(row=5, column=0, pady=5)

        self.rotate_entry = tk.Entry(root, width=10)
        self.rotate_entry.grid(row=6, column=0)

        self.rotate_button = tk.Button(root, text="Rotate", command=self.rotate_image)
        self.rotate_button.grid(row=7, column=0, pady=5)

        self.message_label = tk.Label(root, text="", fg="red")
        self.message_label.grid(row=4, column=0, columnspan=3)

        self.error_label = tk.Label(root, text="", fg="green")
        self.error_label.grid(row=5, column=0, columnspan=3)

        self.gray_lower_label = tk.Label(root, text="Gray Lower")
        self.gray_lower_label.grid(row=6, column=2)
        self.gray_lower_entry = tk.Entry(root, width=10)
        self.gray_lower_entry.grid(row=7, column=2)

        self.gray_upper_label = tk.Label(root, text="Gray Upper")
        self.gray_upper_label.grid(row=8, column=2)
        self.gray_upper_entry = tk.Entry(root, width=10)
        self.gray_upper_entry.grid(row=9, column=2)

        self.gray_level_slicing_button = tk.Button(root, text="Gray-level Slicing", command=self.gray_level_slicing)
        self.gray_level_slicing_button.grid(row=11, column=2, pady=5)
        
        self.recover_button = tk.Button(root, text="Recover Image", command=self.recover_image)
        self.recover_button.grid(row=12, column=0, columnspan=3, pady=5)

        self.original_image = None
        self.image = None

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.tif *.tiff")])
        if file_path:
            self.image = Image.open(file_path)
            self.original_image = self.image
            self.display_image()

    def save_image(self):
        if self.image:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg *.jpeg"), ("TIF files", "*.tif *.tiff")])
            if file_path:
                self.image.save(file_path)
                messagebox.showinfo("Save Image", "Image saved successfully!")

    def display_image(self):
        if self.image:
            self.image = self.image.convert("L")  # Convert to grayscale
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

    def adjust_image_linear(self):
        a = float(self.a_entry.get())
        b = float(self.b_entry.get())
        self.display_image_with_adjustment(a, b, 'linear')

    def adjust_image_exponential(self):
        a = float(self.a_entry.get())
        b = float(self.b_entry.get())
        self.display_image_with_adjustment(a, b, 'exponential')

    def adjust_image_logarithmic(self):
        a = float(self.a_entry.get())
        b = float(self.b_entry.get())
        self.display_image_with_adjustment(a, b, 'logarithmic')
        self.message_label.config(text="Logarithmic adjustment applied")
        self.error_label.config(text="hehe")
    def display_image_with_adjustment(self, a, b, method):
        if self.image:
            grayscale_image = self.image.convert("L")
            width, height = grayscale_image.size
            pixels = grayscale_image.load()
            for x in range(width):
                for y in range(height):
                    pixel_value = pixels[x, y]
                    if method == 'linear':
                        new_value = a * pixel_value + b
                    elif method == 'exponential':
                        new_value = exp(a * pixel_value + b)
                    elif method == 'logarithmic':
                        new_value = log(a * pixel_value + b)
                        if new_value < 0:
                            new_value = 0
                    else:
                        raise ValueError("Invalid method")
                    new_value = max(0, min(255, new_value))
                    pixels[x, y] = int(new_value)
            self.photo = ImageTk.PhotoImage(grayscale_image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo
    def zoom_in(self):
        if self.image:
            try:
                a = float(self.a_entry.get())
                b = float(self.b_entry.get())
                width, height = self.image.size
                new_width = int(width * a)
                new_height = int(height * b)
                resized_image = self.image.resize((new_width, new_height), Image.BILINEAR)
                self.image = resized_image
                self.display_image()
                self.message_label.config(text="Image zoomed in successfully")
                self.error_label.config(text="")
            except Exception as e:
                self.error_label.config(text=f"Error: {str(e)}")
        else:
            self.error_label.config(text="No image to zoom")
    
    def shrink(self):
        if self.image:
            try:
                a = float(self.a_entry.get())
                b = float(self.b_entry.get())
                width, height = self.image.size
                new_width = int(width / a)
                new_height = int(height / b)
                resized_image = self.image.resize((new_width, new_height), Image.BILINEAR)
                self.image = resized_image
                self.display_image()
                self.message_label.config(text="Image shrunk successfully")
                self.error_label.config(text="")
            except Exception as e:
                self.error_label.config(text=f"Error: {str(e)}")
        else:
            self.error_label.config(text="No image to shrink")
    def rotate_image(self):
        if self.image:
            try:
                angle = float(self.rotate_entry.get())
                rotated_image = self.image.rotate(angle, resample=Image.BILINEAR, expand=True)
                self.image = rotated_image
                self.display_image()
                self.message_label.config(text=f"Image rotated by {angle} degrees successfully")
                self.error_label.config(text="")
            except Exception as e:
                self.error_label.config(text=f"Error: {str(e)}")
        else:
            self.error_label.config(text="No image to rotate")

    def gray_level_slicing(self):
        # if self.original_image:
        try:
            gray_lower = int(self.gray_lower_entry.get())
            gray_upper = int(self.gray_upper_entry.get())
            grayscale_image = self.image.convert("L")
            pixels = grayscale_image.load()
            width, height = self.image.size

            for i in range(width):
                for j in range(height):
                    pixel_value = pixels[i, j]
                    if gray_lower <= pixel_value and  pixel_value <= gray_upper:
                        pixels[i, j] = pixel_value
                    else:
                        pixels[i, j] = 0

            self.display_image()
        except ValueError:
            self.message_label.config(text="Invalid input for gray levels")

    def recover_image(self):
        if self.original_image:
            self.image = self.original_image.copy()
            self.display_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()
