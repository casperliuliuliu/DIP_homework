import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
image = cv2.imread('/Users/liushiwen/Desktop/大四上/數位影像處理/DIP_homework/image/powerman.jpg', cv2.IMREAD_GRAYSCALE)

# Perform 2D FFT on the original image
f_transform = np.fft.fft2(image)
# print(f_transform)
magnitude = np.abs(f_transform)
# print(magnitude)
phase = np.angle(f_transform)
# print(phase)

# Perform IFFT on the magnitude image and the phase image separately
reconstructed_magnitude_image = np.fft.ifft2(magnitude)
magnitude_image = np.abs(reconstructed_magnitude_image).astype(np.uint8)
print(reconstructed_magnitude_image)
reconstructed_phase_image = np.fft.ifft2(np.exp(1j * phase)).real

# Display the original, magnitude, and phase images
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(reconstructed_magnitude_image, cmap='gray'), plt.title('Reconstructed Magnitude Image')
plt.subplot(133), plt.imshow(reconstructed_phase_image, cmap='gray'), plt.title('Reconstructed Phase Image')
plt.show()
