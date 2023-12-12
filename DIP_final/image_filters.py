import cv2
from PIL import Image, ImageTk
import numpy as np
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
    # img = zoom_in(img, 1, 2)
    # img = zoom_out(img, 2, 4)


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
def 
if __name__ == "__main__":
    print('hello')
    cap = cv2.VideoCapture(0)
    width=640
    height=480
    cap.set(3, width)  # Set width
    cap.set(4, height)  # Set height
    start(cap)
