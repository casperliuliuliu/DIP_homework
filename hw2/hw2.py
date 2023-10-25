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
        
        self.image_b_label = tk.Label(root)
        self.image_b_label.grid(row=0, column=4, columnspan=3, pady=10)

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

        self.current_state = tk.StringVar()
        self.current_state.set('A')

        self.txt_label = tk.Label(root, text="Current State:")
        self.txt_label.grid(row=1, column=2, pady=5)

        self.state_label = tk.Label(root, textvariable=self.current_state)
        self.state_label.grid(row=2, column=2, pady=5)

        self.toggle_button = tk.Button(root, text="Toggle", command=self.toggle_state)
        self.toggle_button.grid(row=3, column=2, pady=5)

        self.white_bar1 = tk.Button(root, text="7x7 arith", command=self.arithmetic_mean7)
        self.white_bar1.grid(row=1, column=3, pady=5)

        self.white_bar2 = tk.Button(root, text="3x3 arith", command=self.arithmetic_mean3)
        self.white_bar2.grid(row=2, column=3, pady=5)

        self.white_bar3 = tk.Button(root, text="7x7 median", command=self.median_filter7)
        self.white_bar3.grid(row=3, column=3, pady=5)

        self.white_bar4 = tk.Button(root, text="3x3 median", command=self.median_filter)
        self.white_bar4.grid(row=4, column=3, pady=5)

        self.spectrum = tk.Button(root, text="2D-FFT", command=self.fft_2d)
        self.spectrum.grid(row=1, column=4, pady=5)

        self.spectrum = tk.Button(root, text="mag and phase", command=self.magnitude_and_phase)
        self.spectrum.grid(row=2, column=4, pady=5)

        self.dft1 = tk.Button(root, text="1 mul", command=self.multiply_image)
        self.dft1.grid(row=1, column=5, pady=5)

        self.dft2 = tk.Button(root, text="2 dft", command=self.compute_dft)
        self.dft2.grid(row=2, column=5, pady=5)

        self.dft3 = tk.Button(root, text="3 conj", command=self.transform_conjugate)
        self.dft3.grid(row=3, column=5, pady=5)

        self.dft4 = tk.Button(root, text="4 inverse", command=self.inverse_dft)
        self.dft4.grid(row=4, column=5, pady=5)

        self.dft5 = tk.Button(root, text="5 real", command=self.multiply_real)
        self.dft5.grid(row=5, column=5, pady=5)

        self.image_a = None
        self.image_b = None
        self.temp_array = None
    def multiply_real(self):
        if self.current_state.get() == 'A':
            image_array = np.array(self.image_a, dtype=np.float32)
        else:
            image_array = np.array(self.image_b, dtype=np.float32)
        image_array = self.temp_array
        height, width = np.shape(image_array)

        # Create a grid of x and y coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Multiply the real part of the IDFT by (-1)x+y
        result_image_array = self.temp_array .real * (-1) * x + y
        
        # Normalize the result if needed
        min_val = np.min(result_image_array)
        max_val = np.max(result_image_array)
        result_image_array = (result_image_array - min_val) / (max_val - min_val) * 255
        
        # Convert the NumPy array back to an image
        self.temp_array = None
        
        if self.current_state.get() == 'A':
            self.image_a = Image.fromarray(result_image_array.astype(np.uint8))
        else:
            self.image_b = Image.fromarray(result_image_array.astype(np.uint8))
        self.display_image()

    def inverse_dft(self):
        if self.current_state.get() == 'A':
            image_array = np.array(self.image_a, dtype=np.float32)
        else:
            image_array = np.array(self.image_b, dtype=np.float32)
        image_array = self.temp_array

        idft_result = np.fft.ifft2(image_array)
        self.temp_array = idft_result
        if self.current_state.get() == 'A':
            self.image_a = Image.fromarray(idft_result.astype(np.uint8))
        else:
            self.image_b = Image.fromarray(idft_result.astype(np.uint8))
        self.display_image()


    def transform_conjugate(self):
        if self.current_state.get() == 'A':
            image_array = np.array(self.image_a, dtype=np.float32)
        else:
            image_array = np.array(self.image_b, dtype=np.float32)
        image_array = self.temp_array
        dft_conjugate = np.fft.ifft2(image_array.conj())
        self.temp_array = dft_conjugate
        # dft_conjugate = np.conjugate(image_array)

        if self.current_state.get() == 'A':
            self.image_a = Image.fromarray(dft_conjugate.astype(np.uint8))
        else:
            self.image_b = Image.fromarray(dft_conjugate.astype(np.uint8))
        self.display_image()

    def compute_dft(self):
        if self.current_state.get() == 'A':
            image_array = np.array(self.image_a, dtype=np.float32)
        else:
            image_array = np.array(self.image_b, dtype=np.float32)
        image_array = self.temp_array
        dft = np.fft.fft2(image_array)
        self.temp_array = dft

        if self.current_state.get() == 'A':
            self.image_a = Image.fromarray(dft.astype(np.uint8))
        else:
            self.image_b = Image.fromarray(dft.astype(np.uint8))
        # self.print_array()
        self.display_image()

    def multiply_image(self):
        if self.current_state.get() == 'A':
            image_array = np.array(self.image_a, dtype=np.float32)
        else:
            image_array = np.array(self.image_b, dtype=np.float32)
        height, width = np.shape(image_array)
        # Create a grid of x and y coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Calculate the result by multiplying the image by (-1)^(x+y)
        result_image_array = np.multiply(image_array, (-1) ** (x + y))
        self.temp_array = result_image_array
        # min_val = np.min(result_image_array)
        # max_val = np.max(result_image_array)
        # result_image_array = (result_image_array - min_val) / (max_val - min_val) * 255

        # Convert the NumPy array back to a PIL image
        if self.current_state.get() == 'A':
            self.image_a = Image.fromarray(result_image_array.astype(np.uint8))
        else:
            self.image_b = Image.fromarray(result_image_array.astype(np.uint8))
        self.display_image()

    def fft_2d(self):
        if self.current_state.get() == 'A':
            image_array = np.array(self.image_a, dtype=np.float32)
        else:
            image_array = np.array(self.image_b, dtype=np.float32)
        # Perform the 2D-FFT
        fft_image = np.fft.fft2(image_array)
        # Shift zero frequency components to the center
        fft_image = np.fft.fftshift(fft_image)
        # Compute the magnitude of the spectrum
        magnitude_spectrum = np.log(np.abs(fft_image) + 1)
        min_mag = np.min(magnitude_spectrum)
        max_mag = np.max(magnitude_spectrum)
        magnitude_spectrum = (magnitude_spectrum - min_mag) / (max_mag - min_mag) * 255
        if self.current_state.get() == 'A':
            self.image_a = Image.fromarray(magnitude_spectrum.astype(np.uint8))
        else:
            self.image_b = Image.fromarray(magnitude_spectrum.astype(np.uint8))
        self.display_image()

    def magnitude_and_phase(self):
        if self.current_state.get() == 'A':
            image_array = np.array(self.image_a, dtype=np.float32)
        else:
            image_array = np.array(self.image_b, dtype=np.float32)

        fft_image = np.fft.fft2(image_array)
        
        # Compute the magnitude and phase
        magnitude = np.abs(fft_image)
        dft_shift = np.fft.fftshift(fft_image)
        phase = np.angle(dft_shift)
        # Create magnitude-only and phase-only images
        magnitude_image = np.fft.ifft2(magnitude)
        # phase_image = np.fft.ifft2(np.exp(1j * phase))
        # Display the magnitude-only image
        magnitude_image = np.abs(magnitude_image).astype(np.uint8)
        self.image_a = Image.fromarray(magnitude_image.astype(np.uint8))
        self.display_image()
        self.toggle_state()

        self.image_b = Image.fromarray(phase.astype(np.uint8))
        self.display_image()

    def toggle_state(self):
        if self.current_state.get() == 'A':
            self.current_state.set('B')
        else:
            self.current_state.set('A')

    def average_filter(self):
        kernel = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        kernel_size = (3, 3)
         # Apply the convolution with the custom kernel
        if self.current_state.get() == 'A':
            self.image_a = self.image_a.filter(ImageFilter.Kernel(kernel_size, kernel, scale=9, offset=0))
        else:
            self.image_b = self.image_b.filter(ImageFilter.Kernel(kernel_size, kernel, scale=9, offset=0))
        self.display_image()

    def median_filter(self):
        if self.current_state.get() == 'A':
            self.image_a = self.image_a.filter(ImageFilter.MedianFilter(size=3))
        else:
            self.image_b = self.image_b.filter(ImageFilter.MedianFilter(size=3))
        self.display_image()

    def print_array(self):
        if self.current_state.get() == 'A':
            image_array = np.array(self.image_a)
        else:
            image_array = np.array(self.image_b)
            
        print(image_array)

    def laplacian_filter(self):
        kernel = [0, -1, 0, -1, 4, -1, 0, -1, 0]
        if self.current_state.get() == 'A':
            image_array = np.array(self.image_a, dtype=np.float32)
            laplacian_image = cv2.Laplacian(image_array, -1, ksize=3)
            # self.image_a = self.image_a.filter(ImageFilter.Kernel((3, 3), kernel, offset=100))
            self.image_a = Image.fromarray(laplacian_image.astype(np.uint8))
        else:
            image_array = np.array(self.image_b, dtype=np.float32)
            laplacian_image = cv2.Laplacian(image_array, -1, ksize=3)
            self.image_b = Image.fromarray(laplacian_image.astype(np.uint8))
            # self.image_b = self.image_b.filter(ImageFilter.Kernel((3, 3), kernel, offset=100))
        
        self.display_image()

    def arithmetic_mean7(self):
        if self.current_state.get() == 'A':
            self.image_a = self.image_a.filter(ImageFilter.BoxBlur(3))
        else:
            self.image_b = self.image_b.filter(ImageFilter.BoxBlur(3))
        self.display_image()

    def arithmetic_mean3(self):
        if self.current_state.get() == 'A':
            self.image_a = self.image_a.filter(ImageFilter.BoxBlur(1))
        else:
            self.image_b = self.image_b.filter(ImageFilter.BoxBlur(1))
        self.display_image()

    def median_filter7(self):
        if self.current_state.get() == 'A':
            self.image_a = self.image_a.filter(ImageFilter.MedianFilter(7))
        else:
            self.image_b = self.image_b.filter(ImageFilter.MedianFilter(7))

        self.display_image()

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.tif *.tiff")])
        if self.current_state.get() == 'A':
            self.image_a = Image.open(file_path)
            self.image_a = self.image_a.convert("L")  # Convert to grayscale
            self.original_image_a = self.image_a.copy() # Backup
        else:
            self.image_b = Image.open(file_path)
            self.image_b = self.image_b.convert("L")  # Convert to grayscale
            self.original_image_b = self.image_b.copy() # Backup
        self.display_image()

    def display_image(self):
        if self.current_state.get() == 'A':
            self.photo = ImageTk.PhotoImage(self.image_a)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo
        else:
            self.photo = ImageTk.PhotoImage(self.image_b)
            self.image_b_label.config(image=self.photo)
            self.image_b_label.image = self.photo
        

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg *.jpeg"), ("TIF files", "*.tif *.tiff")])
        if self.current_state.get() == 'A':
            self.image_a.save(file_path)
        else:
            self.image_b.save(file_path)
        messagebox.showinfo("Save Image", "Image saved successfully!")
    
    def recover_image(self):
        if self.current_state.get() == 'A':
            self.image_a = self.original_image_a.copy()
        else:
            self.image_b = self.original_image_b.copy()
        self.display_image()
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()