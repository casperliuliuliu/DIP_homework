# User Manual for Image Processing Application

## Table of Contents
1. Introduction
2. Installation
3. Getting Started
4. Image Processing Features
   - 4.1 Open an Image
   - 4.2 Save an Image
   - 4.3 Recover the Original Image
   - 4.4 Applying Filters
   - 4.5 Image Transformation
   - 4.6 Fast Fourier Transform (FFT)
5. Troubleshooting
6. Support and Contact

---

## 1. Introduction

Welcome to the Image Processing Application user manual. This application is designed to help you open, process, and manipulate images with various filters and transformations. The application uses Python and the Tkinter library for its graphical user interface, and it leverages the Python Imaging Library (PIL), OpenCV (cv2), and NumPy for image processing tasks.

This user manual will guide you through the installation process and provide step-by-step instructions for using the features of the application.

## 2. Installation

Before you can use the Image Processing Application, you need to make sure you have the necessary libraries and packages installed. You can install these libraries using pip, the Python package manager.

To install the required libraries, open your terminal or command prompt and run the following commands:

```bash
pip install tkinter
pip install pillow
pip install opencv-python
pip install numpy
```

Once you've installed the necessary packages, you're ready to start using the application.

## 3. Getting Started

To get started with the Image Processing Application, follow these steps:

1. Open a terminal or command prompt.

2. Navigate to the directory where you have saved the application code.

3. Run the Python script containing the application code, for example:

```bash
python image_processing_app.py
```

4. The application's graphical user interface will open, and you'll see various buttons and labels.

## 4. Image Processing Features

The Image Processing Application offers several features for opening, processing, and transforming images. Here's a detailed explanation of each feature:

### 4.1 Open an Image

- Click the "Open Image" button.
- A file dialog will appear, allowing you to select an image file (JPEG or TIF).
- Once you've selected an image, it will be displayed in the application.

### 4.2 Save an Image

- Click the "Save Image" button.
- A file dialog will appear, allowing you to choose the format and location to save the processed image.
- After selecting a location and format, click the "Save" button.

### 4.3 Recover the Original Image

- Click the "Recover Image" button.
- This action will restore the original image, as it was when initially opened.

### 4.4 Applying Filters

#### Average Filter
- The application provides two options for applying an average filter: 7x7 and 3x3.
- Click either "7x7 arith" or "3x3 arith" to apply the corresponding filter.

#### Median Filter
- The application provides two options for applying a median filter: 7x7 and 3x3.
- Click either "7x7 median" or "3x3 median" to apply the corresponding filter.

#### Laplacian Filter
- Click the "Lap Filter" button to apply a Laplacian filter.

### 4.5 Image Transformation

#### Image Transformation
- You can perform various transformations on the image.
- Use the "Toggle" button to switch between two images for transformation.

### 4.6 Fast Fourier Transform (FFT)

#### Fast Fourier Transform (FFT)
- The application allows you to perform a 2D-FFT on the image.
- Click the "2D-FFT" button to apply the transformation.

#### Magnitude and Phase
- Click the "mag and phase" button to compute and display the magnitude and phase of the FFT.

#### Image DFT and IDFT
- The application provides multiple options for image multiplication.
- Use the buttons labeled "1 mul," "2 dft," "3 conj," "4 inverse," and "5 real" for different image DFTs.

## 5. Troubleshooting

If you encounter any issues while using the Image Processing Application, consider the following:

- Ensure that you have installed the required libraries as mentioned in the installation section.
- Verify that the image you're trying to open is in a supported format (JPEG or TIF).

## 6. Support and Contact

If you need further assistance or have questions about the Image Processing Application, you can contact the application developer at [https://github.com/casperliuliuliu].

Enjoy using the Image Processing Application for your image manipulation needs!