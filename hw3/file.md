# User Manual for Image Processing Application

### Run code:
```bash
python hw3.py
```

### Introduction
* Show Red: Displays the red channel of the image.
- Output channel 0, which is red.

* Show Green: Displays the green channel of the image.
- Output channel 1, which is green.

* Show Blue: Displays the blue channel of the image.
- Output channel 2, which is blue.

* Show Hue: Displays the hue channel of the image.
- Turn the Image into HSV format, and output channel 0, which is hue.

* Show Saturation: Displays the saturation channel of the image.
- Turn the Image into HSV format, and output channel 1, which is saturation.

* Show Intensity: Displays the intensity channel of the image.
- Turn the Image into HSV format, and output channel 2, which is intensity.

* Color Complement: Displays the color complement of the image.
- With RGB image, calculate 255 - pixel value through every pixel in 3 channels.
- Output the result.

* Show Histogram: Displays the histogram of the image.
- Calculate the histogram in each channel of RGB image and HSV image by looping over each channel.
- Output 6 histogram, with Red, Green, Blue, Hue, Saturation, Intensity.

* RGB Equalization: Performs RGB equalization on the image.
- Calculate the cumulative distribution function (CDF)
- Normalize the CDF to the range [0, 255]
- Interpolate the CDF values to equalize the channel
- Convert the channel to an unsigned 8-bit integer

* Smooth and Sharp: Performs smoothing and sharpening operations on the image.
- Apply smoothing using a custom 5x5 kernel
- smoothing_kernel = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]]) / 25.0
- Apply the Laplacian filter manually to the RGB image
- laplacian_kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
- Calculate the difference as the absolute difference between pixel values
- Create a PIL Image from the difference array

* Feather segmentation: Performs feather segmentation on the image.
- Define color range thresholds for blue
- Lower bound for hues = 180
- Upper bound for hues = 225
- Lower bound for saturation = 50
- Upper bound for saturation = 185
- Create masks for hue and saturation based on the color range
- Combine the masks using logical AND

### Notice:
- Please use Recover Image button before any other operations. Otherwise, you might not see the image as expected.
