import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFilter
from math import exp, log
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Func3:
    def __init__(self, tab3_frame, main_app):
        self.tab3_frame = tab3_frame
        self.main_app = main_app
        self.create_tab3_widgets()  # Create buttons and labels for Tab 1

    def create_tab3_widgets(self):

        # Image field
        self.image_label = tk.Label(self.tab3_frame)
        self.open_button = tk.Button(self.tab3_frame, text="Open Image", command=self.open_image)
        self.save_button = tk.Button(self.tab3_frame, text="Save Image", command=self.save_image)
        self.recover_button = tk.Button(self.tab3_frame, text="Recover Image", command=self.recover_image)
        self.message_label = tk.Label(self.tab3_frame, text="", fg="green")
        self.error_label = tk.Label(self.tab3_frame, text="", fg="red")
        # column 0
        self.image_label.grid(row=0, column=0, columnspan=3, pady=10)
        self.open_button.grid(row=1, column=0, pady=5)
        self.save_button.grid(row=2, column=0, pady=5)
        self.recover_button.grid(row=3, column=0, pady=5)
        self.message_label.grid(row=11, column=0, columnspan=3)
        self.error_label.grid(row=12, column=0, columnspan=3)

        self.red_button = tk.Button(self.tab3_frame, text="Show Red", command=self.red_part)
        self.red_button.grid(row=1, column=1, pady=5)

        self.green_button = tk.Button(self.tab3_frame, text="Show Green", command=self.green_part)
        self.green_button.grid(row=2, column=1, pady=5)

        self.blue_button = tk.Button(self.tab3_frame, text="Show Blue", command=self.blue_part)
        self.blue_button.grid(row=3, column=1, pady=5)

        self.hue_button = tk.Button(self.tab3_frame, text="Show Hue", command=self.hue_part)
        self.hue_button.grid(row=1, column=2, pady=5)

        self.sat_button = tk.Button(self.tab3_frame, text="Show Saturation", command=self.sat_part)
        self.sat_button.grid(row=2, column=2, pady=5)

        self.gray_button = tk.Button(self.tab3_frame, text="Show Intensity", command=self.gray_part)
        self.gray_button.grid(row=3, column=2, pady=5)

        self.complement_button = tk.Button(self.tab3_frame, text="Color Complement", command=self.color_complement)
        self.complement_button.grid(row=1, column=3, pady=5)
        
        self.show_hist_button = tk.Button(self.tab3_frame, text="Show Histogram", command=self.display_histograms)
        self.show_hist_button.grid(row=2, column=3, pady=5)

        self.rgb_equalization = tk.Button(self.tab3_frame, text="RGB Equalization", command=self.equalization)
        self.rgb_equalization.grid(row=3, column=3, pady=5)

        self.ss_button = tk.Button(self.tab3_frame, text="Smooth and Sharp", command=self.smooth_sharp)
        self.ss_button.grid(row=1, column=4, pady=5)

        self.blue_segmentation = tk.Button(self.tab3_frame, text="Feather segmentation", command=self.segmentation)
        self.blue_segmentation.grid(row=2, column=4, pady=5)
        # Initializing
        self.original_image = None
        self.image = None

    def segmentation(self):
        hsi_image = self.image.convert("HSV")
        # Define color range thresholds for blue
        lower_hue = 180  # Lower bound for blue hues
        upper_hue = 225  # Upper bound for blue hues
        lower_saturation = 50  # Lower bound for saturation
        upper_saturation = 185  # Lower bound for saturation

        # Create masks for hue and saturation based on the color range
        hue_mask = self.create_hue_mask(hsi_image, lower_hue, upper_hue)
        saturation_mask = self.create_saturation_mask(hsi_image, lower_saturation, upper_saturation)

        # Combine the masks using logical OR or arithmetic multiplication
        # You can choose either method based on your specific needs
        # Combined_mask = hue_mask | saturation_mask  # Using logical OR
        # combined_mask = hue_mask * saturation_mask  # Using arithmetic multiplication
        image_array = np.array(self.image)
        image_array[:,:,0] *= saturation_mask
        image_array[:,:,1] *= saturation_mask
        image_array[:,:,2] *= saturation_mask
        sat_image = Image.fromarray(image_array)

        image_array = np.array(self.image)
        image_array[:,:,0] *= hue_mask
        image_array[:,:,1] *= hue_mask
        image_array[:,:,2] *= hue_mask
        hue_image = Image.fromarray(image_array)

        image_array = np.array(self.image)
        combined_mask = hue_mask * saturation_mask  # Using arithmetic multiplication
        image_array[:,:,0] *= combined_mask
        image_array[:,:,1] *= combined_mask
        image_array[:,:,2] *= combined_mask
        combine_image = Image.fromarray(image_array)

        window = tk.Toplevel()
        # Create labels to display the images
        label1 = tk.Label(window, text="After Saturation Mask")
        label2 = tk.Label(window, text="After Hue Mask")
        label3 = tk.Label(window, text="After Logical Combination (AND)")
        label4 = tk.Label(window, text="Original")
        # # Display text labels in row 0 of each column
        label1.grid(row=0, column=0)
        label2.grid(row=0, column=1)
        label3.grid(row=0, column=2)
        label4.grid(row=2, column=0)
        # Convert the images to PhotoImage objects and display them in labels
        photo1 = ImageTk.PhotoImage(sat_image)
        photo2 = ImageTk.PhotoImage(hue_image)
        photo3 = ImageTk.PhotoImage(combine_image)
        photo4 = ImageTk.PhotoImage(self.image)
        # label1.config(image=photo1)        
        image_label1 = tk.Label(window, image=photo1)
        image_label2 = tk.Label(window, image=photo2)
        image_label3 = tk.Label(window, image=photo3)
        image_label4 = tk.Label(window, image=photo4)

        image_label1.photo = photo1
        image_label2.photo = photo2
        image_label3.photo = photo3
        image_label4.photo = photo4
        # Display image labels in row 1 of each column
        image_label1.grid(row=1, column=0)
        image_label2.grid(row=1, column=1)
        image_label3.grid(row=1, column=2)
        image_label4.grid(row=3, column=0)
        # Set the title of the window
        window.title("Image Comparison")
        # Apply the combined mask to the original image to segment the blue feathers
        # self.image = self.apply_mask(hsi_image, combined_mask)
        # self.display_image()

    def create_hue_mask(self, hsi_image, lower_hue, upper_hue):
        hue_channel = np.array(hsi_image.getchannel("H"))
        return ((hue_channel >= lower_hue) & (hue_channel <= upper_hue)).astype(np.uint8)

    def create_saturation_mask(self, hsi_image, lower_saturation, upper_saturation):
        saturation_channel = np.array(hsi_image.getchannel("S"))
        return ((saturation_channel >= lower_saturation) & (saturation_channel <= upper_saturation)).astype(np.uint8)

    def apply_mask(self, image, mask):
        mask = Image.fromarray(mask, "L")
        return Image.composite(image, Image.new("HSV", image.size, (0, 0, 0)), mask)

    def smooth_sharp(self):

        # Convert the original image to RGB
        rgb_image = self.image.convert("RGB")
        hsv_image = self.image.convert("HSV")
        smoothing_kernel = np.array([[1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1]]) / 25.0
        # Apply smoothing using a custom 5x5 kernel
        smoothed_rgb = self.apply_filter(rgb_image, smoothing_kernel)
        smoothed_hsv = self.apply_filter(hsv_image, smoothing_kernel)
        
        # Apply the Laplacian filter manually to the RGB image
        laplacian_kernel = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]])
        sharpened_rgb = self.apply_filter(smoothed_rgb, laplacian_kernel)
        sharpened_hsv = self.apply_filter(hsv_image, laplacian_kernel)
        

        # Calculate the difference as the absolute difference between pixel values
        difference_array_rgb = np.abs(np.array(rgb_image) - np.array(sharpened_rgb))
        difference_array_hsv = np.abs(np.array(rgb_image) - np.array(sharpened_rgb))
        # Create a PIL Image from the difference array
        difference_image_rgb = Image.fromarray(np.uint8(difference_array_rgb))
        difference_image_hsv = Image.fromarray(np.uint8(difference_array_hsv))

        window = tk.Toplevel()
        # Create labels to display the images
        label1 = tk.Label(window, text="Smooth RGB")
        label2 = tk.Label(window, text="Sharpen RGB")
        label3 = tk.Label(window, text="Difference RGB")
        label4 = tk.Label(window, text="Smooth HSV")
        label5 = tk.Label(window, text="Sharpen HSV")
        label6 = tk.Label(window, text="Difference HSV")
        # # Display text labels in row 0 of each column
        label1.grid(row=0, column=0)
        label2.grid(row=0, column=1)
        label3.grid(row=0, column=2)
        label4.grid(row=2, column=0)
        label5.grid(row=2, column=1)
        label6.grid(row=2, column=2)
        # Convert the images to PhotoImage objects and display them in labels
        photo1 = ImageTk.PhotoImage(smoothed_rgb)
        photo2 = ImageTk.PhotoImage(sharpened_rgb)
        photo3 = ImageTk.PhotoImage(difference_image_rgb)
        photo4 = ImageTk.PhotoImage(smoothed_hsv)
        photo5 = ImageTk.PhotoImage(sharpened_hsv)
        photo6 = ImageTk.PhotoImage(difference_image_hsv)
        # label1.config(image=photo1)        
        image_label1 = tk.Label(window, image=photo1)
        image_label2 = tk.Label(window, image=photo2)
        image_label3 = tk.Label(window, image=photo3)
        image_label4 = tk.Label(window, image=photo4)
        image_label5 = tk.Label(window, image=photo5)
        image_label6 = tk.Label(window, image=photo6)

        image_label1.photo = photo1
        image_label2.photo = photo2
        image_label3.photo = photo3
        image_label4.photo = photo4
        image_label5.photo = photo5
        image_label6.photo = photo6
        # Display image labels in row 1 of each column
        image_label1.grid(row=1, column=0)
        image_label2.grid(row=1, column=1)
        image_label3.grid(row=1, column=2)
        image_label4.grid(row=3, column=0)
        image_label5.grid(row=3, column=1)
        image_label6.grid(row=3, column=2)
        # Set the title of the window
        window.title("Image Comparison")

        self.display_image()

    def apply_filter(self, image, kernel):
        image_array = np.array(image)
        height, width = image_array.shape[0], image_array.shape[1]
        kernel_size = len(kernel)
        pad = kernel_size // 2
        result = np.zeros_like(image_array, dtype=np.uint8)
        
        for i in range(pad, height - pad):
            for j in range(pad, width - pad):
                for c in range(3):
                    result[i, j, c] = np.sum(image_array[i-pad:i+pad+1, j-pad:j+pad+1, c] * kernel)

        if kernel_size == 3:
            result = image_array + 1.0 * result
            result *= 255.0/np.amax(result)
            # Clip the values to the valid range [0, 255]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result)

    
    def histogram_equalization(self, channel):
        # Calculate the histogram of the channel
        histogram, bins = np.histogram(channel, bins=256, range=(0, 256))
        
        # Calculate the cumulative distribution function (CDF)
        cdf = histogram.cumsum()
        
        # Normalize the CDF to the range [0, 255]
        cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        
        # Interpolate the CDF values to equalize the channel
        equalized_channel = np.interp(channel, bins[:-1], cdf)
        
        # Convert the channel to an unsigned 8-bit integer
        equalized_channel = equalized_channel.astype(np.uint8)
        
        return equalized_channel
    
    def equalization(self):
        rgb_array = [np.array(ii) for ii in self.image.split()]
        equalized_channels = [self.histogram_equalization(channel) for channel in rgb_array]
    
        # Merge the equalized channels back into an RGB image
        self.image = Image.merge("RGB", [Image.fromarray(channel) for channel in equalized_channels])
        window = tk.Toplevel()
    
        # Create labels to display the images
        label1 = tk.Label(window, text="Equalized Image")
        label2 = tk.Label(window, text="Original Image")

        # # Display text labels in row 0 of each column
        label1.grid(row=0, column=0)
        label2.grid(row=0, column=1)
        
        # Convert the images to PhotoImage objects and display them in labels
        photo1 = ImageTk.PhotoImage(self.image)
        photo2 = ImageTk.PhotoImage(self.original_image)
        # label1.config(image=photo1)        
        image_label1 = tk.Label(window, image=photo1)
        image_label2 = tk.Label(window, image=photo2)

        image_label1.photo = photo1
        image_label2.photo = photo2

        # Display image labels in row 1 of each column
        image_label1.grid(row=1, column=0)
        image_label2.grid(row=1, column=1)
        
        # Set the title of the window
        window.title("Image Comparison")

        self.display_image()

    def display_histograms(self):

        # Convert the PIL image to a NumPy array
        rgb_array = np.array(self.image)

        # Create an HSI version of the image
        hsi_image = self.image.convert("HSV")
        hsi_array = np.array(hsi_image)

        # Create a pop-up window for the histograms
        histogram_window = tk.Toplevel()
        histogram_window.title("Histograms")

        # Create six frames for displaying histograms
        rgb_histogram_frames = [ttk.Frame(histogram_window) for _ in range(3)]
        hsi_histogram_frames = [ttk.Frame(histogram_window) for _ in range(3)]
        tag = ["RED", "GREEN", "BLUE", "HUE", "SATURATION", "INTENSITY"]
        # Create and display RGB histograms
        for i in range(3):
            rgb_histogram = self.create_histogram(rgb_array, i)
            self.display_histogram(rgb_histogram, rgb_histogram_frames[i], f"{tag[i]}", i)

        # Create and display HSI histograms
        for i in range(3):
            hsi_histogram = self.create_histogram(hsi_array, i)
            self.display_histogram(hsi_histogram, hsi_histogram_frames[i], f"{tag[i+3]}", i + 3)

    def create_histogram(self, image, channel_index):
        # Calculate the histogram for the specified channel
        histogram, _ = np.histogram(image[:, :, channel_index], bins=256, range=(0, 256))
        return image[:, :, channel_index]

    def display_histogram(self, histogram, frame, title, row):
        fig = Figure(figsize=(4, 2), dpi=100)
        ax = fig.add_subplot(111)
        # ax.bar(bin_edges[:-1], histogram, width=16, align='edge', color='black', alpha=0.7)
        ax.hist(histogram.flatten(), bins=256, range=(0,255))
        ax.set_xlim(0, 256)

        # Create a canvas to display the histogram
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack()
        canvas.draw()

        # Add a title to the histogram
        tk.Label(frame, text=title).pack()

        # Place the frame in the pop-up window
        frame.grid(row=row%3, column=int(row/3))

    def color_complement(self):
        self.image = ImageOps.invert(self.image)
        self.display_image()

    def rgb2hsv(self):
        self.image = self.image.convert("HSV")
        self.display_image()

    def hue_part(self):
        self.rgb2hsv()
        self.image = self.image.split()[0]
        self.display_image()

    def sat_part(self):
        self.rgb2hsv()
        self.image = self.image.split()[1]
        self.display_image()

    def gray_part(self):
        self.rgb2hsv()
        self.image = self.image.split()[2]
        self.display_image()

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
            
