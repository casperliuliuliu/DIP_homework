def __init__(self, root, video_source=0, width=640, height=480):##separate page
        # ... (existing code)

        self.btn_frame_additional = tk.Frame(root)  # Additional button frame
        self.btn_frame_additional.pack(pady=10)

        self.btn_additional = ttk.Button(self.btn_frame_additional, text="Additional Button", command=self.additional_button_action)
        self.btn_additional.grid(row=0, column=0, padx=10)

        self.btn_show_additional_page = ttk.Button(self.btn_frame, text="Show Additional Page", command=self.show_additional_page)
        self.btn_show_additional_page.grid(row=0, column=5, padx=10)

        # Additional page
        self.additional_page = None

    def additional_button_action(self):
        print("Additional button clicked!")

    def show_additional_page(self):
        if self.additional_page is None:
            self.additional_page = tk.Toplevel(self.root)
            self.additional_page.title("Additional Page")

            btn_additional_page = ttk.Button(self.additional_page, text="Close Additional Page", command=self.close_additional_page)
            btn_additional_page.pack(pady=10)

    def close_additional_page(self):
        if self.additional_page:
            self.additional_page.destroy()
            self.additional_page = None
----------------------------------------------------------------------------------------------------------------------
 def avg_filter(self, img):
        kernel_size = 3  # You can adjust this based on your requirement
        padding = (kernel_size - 1) // 2

        # Apply zero-padding to the image
        img_padded = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

        # Create the average filter kernel
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

        # Apply the convolution operation
        img_filtered = cv2.filter2D(img_padded, -1, kernel)

        # Crop the result to remove padding
        img_filtered = img_filtered[padding:-padding, padding:-padding]

        return Image.fromarray(img_filtered)
----------------------------------------------------------------------------------------------------------------------
def median_filter(self, img):
        kernel_size = 3  # You can adjust this based on your requirement
        padding = (kernel_size - 1) // 2

        # Apply zero-padding to the image
        img_padded = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

        # Initialize an empty array for the filtered image
        img_filtered = np.zeros_like(img)

        for i in range(padding, img_padded.shape[0] - padding):
            for j in range(padding, img_padded.shape[1] - padding):
                # Extract the neighborhood
                neighborhood = img_padded[i - padding:i + padding + 1, j - padding:j + padding + 1]

                # Apply median filter to each channel separately
                for channel in range(img.shape[2]):
                    img_filtered[i - padding, j - padding, channel] = np.median(neighborhood[:, :, channel])

        return Image.fromarray(img_filtered)
------------------------------------------------------------------------------------------------------------------------------
def TwoDimension_FFT_Amplitude(self, img_array):
        f = np.fft.fft2(img_array)
        fshift = 1 + np.fft.fftshift(f)
        spectrum = 20 * np.log(np.abs(fshift))
        sep = np.abs(f)
        imgsep = np.real(np.fft.ifft2(sep))

        return spectrum, imgsep

    def TwoDimension_FFT_Phase(self, img_array):
        f = np.fft.fft2(img_array)
        fshift = np.fft.fftshift(f)
        spectrum = np.angle(fshift)
        sep = np.exp(1j * np.angle(f))
        imgsep = np.real(np.fft.ifft2(sep))

        return spectrum, imgsep

    def show_fft_amplitude_page(self):
        amplitude_page = Toplevel(self.root)
        amplitude_page.title("FFT Amplitude")

        img = cv2.cvtColor(cv2.imread("path/to/your/image.jpg"), cv2.COLOR_BGR2GRAY)
        spectrum, imgsep = self.TwoDimension_FFT_Amplitude(img)

        spectrum_label = Label(amplitude_page)
        spectrum_label.pack()

        img_spectrum = Image.fromarray(spectrum)
        img_spectrum = ImageTk.PhotoImage(img_spectrum)
        spectrum_label.config(image=img_spectrum)
        spectrum_label.image = img_spectrum

        imgsep_label = Label(amplitude_page)
        imgsep_label.pack()

        img_sep = Image.fromarray(imgsep)
        img_sep = ImageTk.PhotoImage(img_sep)
        imgsep_label.config(image=img_sep)
        imgsep_label.image = img_sep

    def show_fft_phase_page(self):
        phase_page = Toplevel(self.root)
        phase_page.title("FFT Phase")

        img = cv2.cvtColor(cv2.imread("path/to/your/image.jpg"), cv2.COLOR_BGR2GRAY)
        spectrum, imgsep = self.TwoDimension_FFT_Phase(img)

        spectrum_label = Label(phase_page)
        spectrum_label.pack()

        img_spectrum = Image.fromarray(spectrum)
        img_spectrum = ImageTk.PhotoImage(img_spectrum)
        spectrum_label.config(image=img_spectrum)
        spectrum_label.image = img_spectrum

        imgsep_label = Label(phase_page)
        imgsep_label.pack()

        img_sep = Image.fromarray(imgsep)
        img_sep = ImageTk.PhotoImage(img_sep)
        imgsep_label.config(image=img_sep)
        imgsep_label.image = img_sep
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def display_bit_plane(self):
        try:
            bit_plane = self.bit_plane_var.get()
            image_array = np.array(self.image)
            bit_plane_image = ((image_array >> bit_plane) & 1) * 255
            bit_plane_image = Image.fromarray(bit_plane_image)

            # Assuming self.image is a tkinter Label or Canvas where you display the image
            # If not, replace self.image with the appropriate widget reference
            self.image.config(image=ImageTk.PhotoImage(bit_plane_image))
            self.image.image = ImageTk.PhotoImage(bit_plane_image)
        except Exception as e:
            # Display an error message if an exception occurs
            self.show_error_message(f"Error: {str(e)}")
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def color_complement(self, image_array):
        # Get the shape of the image array
        height, width, _ = image_array.shape

        # Create a new array for the color complement
        complement_image_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Perform color complement for each pixel
        complement_image_array[:, :, 0] = 255 - image_array[:, :, 0]  # Red component
        complement_image_array[:, :, 1] = 255 - image_array[:, :, 1]  # Green component
        complement_image_array[:, :, 2] = 255 - image_array[:, :, 2]  # Blue component

        return complement_image_array

def display_color_complement(self):
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            complement_image_array = self.color_complement(img)

            # Assuming self.canvas_left is a tkinter Canvas where you display the image
            # If not, replace self.canvas_left with the appropriate widget reference
            complement_image = Image.fromarray(complement_image_array)
            complement_image = ImageTk.PhotoImage(complement_image)
            self.canvas_left.config(width=complement_image.width(), height=complement_image.height())
            self.canvas_left.create_image(0, 0, anchor=tk.NW, image=complement_image)
            self.canvas_left.image = complement_image  # Save reference to prevent garbage collection

        except Exception as e:
            # Display an error message if an exception occurs
            self.show_error_message(f"Error: {str(e)}")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------