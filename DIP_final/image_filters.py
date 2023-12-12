import cv2
from PIL import Image, ImageTk
import numpy as np
last_img = 0
def start(cap):
    while(True):
        # 擷取影像
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # 彩色轉灰階
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_img = transform(frame)

        # 顯示圖片
        cv2.imshow('live', new_img)

        # 按下 q 鍵離開迴圈
        if cv2.waitKey(1) == ord('q'):
            break

def transform(img):
    """
    Test the function to do the filter.
    """
    global last_img
    # img = zoom_in(img, 1, 2)
    # img = zoom_out(img, 2, 4)
    # img = flip_right_to_left(img)
    # img = flip_top_to_down(img)
    # img = show_image_difference(img)
    
    return img

def zoom_in(img, a, b):
    # Get the original image size
    original_height, original_width, _ = img.shape

    # Resize the cropped image back to the original size
    processed_img = cv2.resize(img, (original_width*a, original_height*b))

    return processed_img

def zoom_out(img, a, b):
    # Get the original image size
    original_height, original_width, _ = img.shape

    # Calculate the region to crop (1/2 of the original size)
    new_width = original_width // a  # Crop from 1/4 to 3/4 of the width
    new_height = original_height // b  # Crop from 1/4 to 3/4 of the height

    # Resize the cropped image back to the original size
    processed_img = cv2.resize(img, (new_width, new_height))

    return processed_img

def flip_right_to_left(img):
    flipped_img = cv2.flip(img, 1)  # 1 denotes horizontal flip

    return flipped_img

def flip_top_to_down(img):
    flipped_img = cv2.flip(img, 0)  # 1 denotes horizontal flip

    return flipped_img

def show_image_difference(current_img):
    global last_img
    # Ensure the images have the same size
    if isinstance(last_img, int):
        last_img = current_img

        return current_img
    
    if current_img.shape != last_img.shape:
        raise ValueError("Image sizes must match.")

    # Compute the absolute difference between the images
    diff_img = cv2.absdiff(current_img, last_img)
    last_img = current_img

    return diff_img
if __name__ == "__main__":
    print('hello')
    cap = cv2.VideoCapture(0)
    width=640
    height=480
    cap.set(3, width)  # Set width
    cap.set(4, height)  # Set height
    start(cap)
