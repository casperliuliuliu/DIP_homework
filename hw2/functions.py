from PIL import Image
import cv2
import numpy as np

# Open the image using Pillow (PIL)
pil_image = Image.open('/Users/liushiwen/Desktop/大四上/數位影像處理/DIP_homework/hw2/Images/pirate_a.tif')  # Replace with the path to your image

# Convert the Pillow image to a NumPy array
pil_image = np.array(pil_image)

# Convert the image data type to uint8 (required for OpenCV)
def pil_turnto_cv2(pil_image):
    pil_image = pil_image.astype(np.uint8)
    # Convert the NumPy array to a cv2 image
    cv2_image = cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)
    return cv2_image
# Now, you can use cv2_image as a cv2 image object

# Display the cv2 image (for example)
cv2.imshow('OpenCV Image', cv2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
