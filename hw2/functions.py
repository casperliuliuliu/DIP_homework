from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFilter
from math import exp, log
import numpy as np
class ImageEditorFunctions:
def open_image(self):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.tif *.tiff")])
    try:
        self.image = Image.open(file_path)
        self.image = self.image.convert("L")  # Convert to grayscale
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

def adjust_image_linear(self):
    self.display_image_with_adjustment('linear')
    self.message_label.config(text="Linear adjustment applied")

def adjust_image_exponential(self):
    self.display_image_with_adjustment('exponential')
    self.message_label.config(text="Exponential adjustment applied")

def adjust_image_logarithmic(self):
    self.display_image_with_adjustment('logarithmic')
    self.message_label.config(text="Logarithmic adjustment applied")

def display_image_with_adjustment(self, method):
    try:
        a = float(self.a_entry.get())
        b = float(self.b_entry.get())
        width, height = self.image.size
        pixels = self.image.load() # Turn image data into pixel data
        
        # Loop through each pixel, recalculate pixel value
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
                new_value = max(0, min(255, new_value)) # In case overflow
                pixels[x, y] = int(new_value) # Image data also changed
        self.display_image()
    except Exception as e:
        self.error_label.config(text=f"Error: {str(e)}")

def zoom_in(self):
    self.resizing("zoom_in")
    self.message_label.config(text="Image zoomed in successfully")
    
def shrink(self):
    self.resizing("shrink")
    self.message_label.config(text="Image shrunk successfully")

def resizing(self, method):
    try:
        a = float(self.a_entry.get())
        b = float(self.b_entry.get())
        width, height = self.image.size

        # Calculate new w and h.
        if method == "zoom_in":
            new_width = int(width * a)
            new_height = int(height * b)

        elif method == "shrink":
            new_width = int(width / a)
            new_height = int(height / b)

        resized_image = self.image.resize((new_width, new_height), Image.BILINEAR) # Bilinear interpolation
        self.image = resized_image
        self.display_image()
        self.error_label.config(text="")
    except Exception as e:
        self.error_label.config(text=f"Error: {str(e)}")

def rotate_image(self):
    try:
        angle = float(self.rotate_entry.get())
        rotated_image = self.image.rotate(angle, resample=Image.BILINEAR, expand=True)
        self.image = rotated_image
        self.display_image()
        self.message_label.config(text=f"Image rotated by {angle} degrees successfully")
        self.error_label.config(text="")
    except Exception as e:
        self.error_label.config(text=f"Error: {str(e)}")

def gray_level_slicing(self):
    try:
        gray_lower = int(self.gray_lower_entry.get())
        gray_upper = int(self.gray_upper_entry.get())
        pixels = self.image.load()
        width, height = self.image.size # Turn image data into pixel data

        # Loop through each pixel, set pixel out of range into 0
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

def display_histogram(self):
    try:
        histogram = self.image.histogram()
        grayscale_histogram = histogram[0:256]

        # Calculate the maximum count for normalization
        # max_count = max(grayscale_histogram)
        max_count = self.image.width * self.image.height

        # Create a blank image for the histogram
        hist_image = Image.new('RGB', (300, 200), "white")
        draw = ImageDraw.Draw(hist_image)
        # Draw histogram bars
        for i in range(256):
            # print(grayscale_histogram[i])
            bar_height = int(2000 * grayscale_histogram[i] / max_count)
            draw.rectangle([i+20, 200 - bar_height, i + 21, 200], fill="black")

        # Convert PIL image to PhotoImage
        hist_photo = ImageTk.PhotoImage(hist_image)

        # Display the histogram image on the Canvas
        self.histogram_canvas.create_image(0, 0, anchor=tk.NW, image=hist_photo)
        self.histogram_canvas.hist_photo = hist_photo

    except Exception as e:
        self.error_label.config(text=f"Error: {str(e)}")
def auto_level(self):
    try:
        # Grayscale image auto-level
        self.image = self.auto_level_single_channel(self.image)
        self.display_image()
    except Exception as e:
        self.error_label.config(text=f"Error: {str(e)}")

def auto_level_single_channel(self, image):
    # Histogram equalization
    image = ImageOps.equalize(image)
    return image

def display_bit_plane(self, *args):
    self.recover_image()
    try:
        bit_plane = self.bit_plane_var.get()
        image_array = np.array(self.image)
        bit_plane_image = ((image_array >> bit_plane) & 1) * 255
        bit_plane_image = Image.fromarray(bit_plane_image)
        self.image = bit_plane_image
        self.display_image()
    except Exception as e:
        self.error_label.config(text=f"Error: {str(e)}")
        
def apply_smoothing(self):
    try:
        smoothing_level = self.smoothing_var.get()
        smoothed_image = self.image.filter(ImageFilter.GaussianBlur(smoothing_level))
        self.image = smoothed_image
        self.display_image()
    except Exception as e:
        self.error_label.config(text=f"Error: {str(e)}")

def apply_sharpening(self):
    try:
        sharpening_level = self.sharpening_var.get()
        sharpening_level *= 100 # Turn Var into percentage
        sharpening_level = int(sharpening_level)

        sharpened_image = self.image.filter(ImageFilter.UnsharpMask(radius=2, percent=sharpening_level))
        self.image = sharpened_image
        self.display_image()
    except Exception as e:
        self.error_label.config(text=f"Error: {str(e)}")
