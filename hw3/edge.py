import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('/Users/liushiwen/Desktop/大四上/數位影像處理/DIP_homework/image/powerman.jpg', cv2.IMREAD_GRAYSCALE)

# Define the Laplacian operators
laplacian_operator_0 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
laplacian_operator_0 = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=np.float32)
laplacian_operator_1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
laplacian_operator_2 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
laplacian_operator_3 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
laplacian_operator_4 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

# Apply the Laplacian operators
laplacian_response_0 = cv2.filter2D(image, -1, laplacian_operator_0)
laplacian_response_1 = cv2.filter2D(image, -1, laplacian_operator_1)
laplacian_response_2 = cv2.filter2D(image, -1, laplacian_operator_2)
laplacian_response_3 = cv2.filter2D(image, -1, laplacian_operator_3)
laplacian_response_4 = cv2.filter2D(image, -1, laplacian_operator_4)

# Display the original image and the Laplacian responses
plt.figure(figsize=(24, 8))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(laplacian_response_0, cmap='gray'), plt.title('Laplacian Operator #1')
plt.subplot(133), plt.imshow(laplacian_response_4, cmap='gray'), plt.title('Laplacian Operator #2')
# plt.subplot(134), plt.imshow(laplacian_response_3, cmap='gray'), plt.title('Laplacian Operator #2')
# plt.subplot(135), plt.imshow(laplacian_response_4, cmap='gray'), plt.title('Laplacian Operator #2')
plt.show()
