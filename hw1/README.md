# Image Editor App Readme

## Introduction
The Image Editor App is a graphical user interface (GUI) application built using Python and the tkinter library for image processing. It allows users to open, edit, and save grayscale images. The app provides various image editing features such as contrast adjustment, resizing, rotation, histogram display, auto-leveling, gray-level slicing, bit-plane slicing, and various filters like smoothing and sharpening.

## Getting Started
To use the Image Editor App, follow these steps:

1. Clone or download the repository containing the source code.
2. Ensure you have Python installed on your system.
3. Install the required Python libraries: tkinter, Pillow (PIL), and numpy. You can install these libraries using pip:

   ```
   pip install pillow numpy
   ```

4. Run the application by executing the Python script. For example:

   ```
   python image_editor.py
   ```

## Using the App

### Opening an Image
- Click the "Open Image" button to select an image file (supported formats: .jpg, .jpeg, .tif, .tiff).
- The selected image will be displayed in the application.

### Saving an Image
- Click the "Save Image" button to save the currently displayed image. You can choose the file format and location to save the image.

### Recovering the Original Image
- Click the "Recover Image" button to revert to the original, unedited image.

### Image Adjustments
- The "a" and "b" values are used for contrast and resizing operations. Enter appropriate values in these fields before applying adjustments.
- Click the "Linear," "Exponential," or "Logarithmic" buttons to apply the respective adjustment to the image.

### Resizing
- Use the "Zoom In" and "Shrink" buttons to resize the image. Adjust the "a" and "b" values for resizing parameters.

### Rotation
- Enter the desired angle in the "Angle (for rotate)" field.
- Click the "Rotate" button to rotate the image by the specified angle.

### Histogram Display
- Click the "Display Histogram" button to display the grayscale histogram of the image. The histogram will be shown in a separate canvas.

### Auto-Leveling
- Click the "Auto-Level" button to perform histogram equalization on the image, improving its contrast.

### Gray-Level Slicing
- Specify the lower and upper gray levels in the "Gray Lower" and "Gray Upper" fields.
- Click the "Gray-level Slicing" button to apply the slicing operation.

### Bit-Plane Slicing
- Use the dropdown menu to select the bit-plane for slicing (ranging from 0 to 7).
- The selected bit-plane will be displayed as a binary image.

### Smoothing
- Adjust the "Smoothing Level" using the slider.
- Click the "Smooth" button to apply a smoothing filter to the image.

### Sharpening
- Adjust the "Sharpening Level" using the slider (in percentage).
- Click the "Sharpen" button to apply a sharpening filter to the image.

## Troubleshooting
- If you encounter any errors or issues while using the Image Editor App, please refer to the error messages displayed on the application or the console for guidance.

## Author
- Casper Liu

## License
This application is provided under an open-source license. You are free to use, modify, and distribute it in accordance with the license terms. Please refer to the "LICENSE" file in the source code repository for details.

Enjoy editing your images with the Image Editor App!