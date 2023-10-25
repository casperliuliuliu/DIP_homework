import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFilter
from math import exp, log
import numpy as np
class Func1:
    def __init__(self, tab1_frame, main_app):
        self.tab1_frame = tab1_frame
        self.main_app = main_app
        self.create_tab1_widgets()  # Create buttons and labels for Tab 1

    def create_tab1_widgets(self):
        # Image field
        self.image_label = tk.Label(self.tab1_frame)
        self.open_button = tk.Button(self.tab1_frame, text="Open Image", command=self.open_image)
        self.save_button = tk.Button(self.tab1_frame, text="Save Image", command=self.save_image)
        self.recover_button = tk.Button(self.tab1_frame, text="Recover Image", command=self.recover_image)
        self.message_label = tk.Label(self.tab1_frame, text="", fg="green")
        self.error_label = tk.Label(self.tab1_frame, text="", fg="red")

        # Ask for the value a
        self.a_label = tk.Label(self.tab1_frame, text="a (for contrast and resize)")
        self.a_entry = tk.Entry(self.tab1_frame, width=10)

        # Ask for the value b
        self.b_label = tk.Label(self.tab1_frame, text="b (for contrast and resize)")
        self.b_entry = tk.Entry(self.tab1_frame, width=10)

        # Contrast and Brightness methods
        self.linear_button = tk.Button(self.tab1_frame, text="Linear", command=self.adjust_image_linear)
        self.exponential_button = tk.Button(self.tab1_frame, text="Exponential", command=self.adjust_image_exponential)
        self.logarithmic_button = tk.Button(self.tab1_frame, text="Logarithmic", command=self.adjust_image_logarithmic)
        
        # Zoom_in and Shrink
        self.zoom_in_button = tk.Button(self.tab1_frame, text="Zoom In", command=self.zoom_in)
        self.shrink_button = tk.Button(self.tab1_frame, text="Shrink", command=self.shrink)

        # Rotation
        self.rotate_label = tk.Label(self.tab1_frame, text="Angle (for rotate)")
        self.rotate_entry = tk.Entry(self.tab1_frame, width=10)
        self.rotate_button = tk.Button(self.tab1_frame, text="Rotate", command=self.rotate_image)

        # Histogram display and Auto-leveling(histogram equalization)
        self.histogram_button = tk.Button(self.tab1_frame, text="Display Histogram", command=self.display_histogram)
        self.auto_level_button = tk.Button(self.tab1_frame, text="Auto-Level", command=self.auto_level)
        # Histogram_canvas
        self.histogram_canvas = tk.Canvas(self.tab1_frame, width=300, height=200)

        # Gray-level slicing
        self.gray_lower_label = tk.Label(self.tab1_frame, text="Gray Lower")
        self.gray_upper_label = tk.Label(self.tab1_frame, text="Gray Upper")
        self.gray_lower_entry = tk.Entry(self.tab1_frame, width=10)
        self.gray_upper_entry = tk.Entry(self.tab1_frame, width=10)
        self.gray_level_slicing_button = tk.Button(self.tab1_frame, text="Gray-level Slicing", command=self.gray_level_slicing)

        # Bit setting
        self.bit_plane_label = tk.Label(self.tab1_frame, text="Bit-Plane:")
        self.bit_plane_var = tk.IntVar()
        self.bit_plane_var.set(7)  # Set default to the 7th bit-plane
        self.bit_plane_dropdown = tk.OptionMenu(self.tab1_frame, self.bit_plane_var, *range(8), command=self.display_bit_plane)

        self.smoothing_label = tk.Label(self.tab1_frame, text="Smoothing Level:")
        self.smoothing_var = tk.DoubleVar()
        self.smoothing_var.set(1.0)  # Set default smoothing level to 1.0
        self.smoothing_slider = tk.Scale(self.tab1_frame, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.smoothing_var)
        self.sharpening_label = tk.Label(self.tab1_frame, text="Sharpening Level:")
        self.sharpening_var = tk.DoubleVar()
        self.sharpening_var.set(1.0)  # Set default sharpening level to 1.0
        self.sharpening_slider = tk.Scale(self.tab1_frame, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.sharpening_var)
        self.apply_smoothing_button = tk.Button(self.tab1_frame, text="Smooth", command=self.apply_smoothing)
        self.apply_sharpening_button = tk.Button(self.tab1_frame, text="Sharpen", command=self.apply_sharpening)

        # column 0
        self.image_label.grid(row=0, column=0, columnspan=3, pady=10)
        self.open_button.grid(row=1, column=0, pady=5)
        self.save_button.grid(row=2, column=0, pady=5)
        self.recover_button.grid(row=3, column=0, pady=5)
        self.message_label.grid(row=11, column=0, columnspan=3)
        self.error_label.grid(row=12, column=0, columnspan=3)
        # column1
        self.rotate_label.grid(row=1, column=1)
        self.rotate_entry.grid(row=2, column=1)
        self.rotate_button.grid(row=3, column=1, pady=5)
        self.zoom_in_button.grid(row=4, column=1, pady=5)
        self.shrink_button.grid(row=5, column=1, pady=5)
        self.histogram_button.grid(row=6, column=1, pady=5)
        self.auto_level_button.grid(row=7, column=1, pady=5)
        # column 2
        self.a_label.grid(row=1, column=2)
        self.a_entry.grid(row=2, column=2)
        self.b_label.grid(row=3, column=2)
        self.b_entry.grid(row=4, column=2)
        self.linear_button.grid(row=5, column=2, pady=5)
        self.exponential_button.grid(row=6, column=2, pady=5)
        self.logarithmic_button.grid(row=7, column=2, pady=5)
        # column 3
        self.gray_lower_label.grid(row=1, column=3)
        self.gray_lower_entry.grid(row=2, column=3)
        self.gray_upper_label.grid(row=3, column=3)
        self.gray_upper_entry.grid(row=4, column=3)
        self.gray_level_slicing_button.grid(row=5, column=3, pady=5)
        # column 4
        self.histogram_canvas.grid(row=0, column=4, columnspan=2, pady=5)
        self.bit_plane_label.grid(row=1, column=4, pady=5)
        self.smoothing_label.grid(row=2, column=4, pady=5)
        self.sharpening_label.grid(row=3, column=4, pady=5)
        self.apply_smoothing_button.grid(row=4, column=4, pady=5)
        # column 5
        self.bit_plane_dropdown.grid(row=1, column=5, pady=5)
        self.smoothing_slider.grid(row=2, column=5, pady=5)
        self.sharpening_slider.grid(row=3, column=5, pady=5)
        self.apply_sharpening_button.grid(row=4, column=5, pady=5)

        # Initializing
        self.original_image = None
        self.image = None

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
            
