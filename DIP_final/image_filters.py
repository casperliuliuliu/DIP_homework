import cv2
from PIL import Image, ImageTk
import numpy as np
import math
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
    # img = enlarge_effect(img)
    img = enlarge_line_effect_fixed(img)
    # img = enlarge_effect_optimized(img)
    # img = reduce_effect_optimized(img)
    # img = apply_wave_effect(img)
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

def enlarge_effect(img, radius = 100):
    h, w, n = img.shape
    cx = w / 2
    cy = h / 2
    r = int(radius / 2.0)
    new_img = img.copy()
    for i in range(w):
        for j in range(h):
            tx = i - cx
            ty = j - cy
            distance = tx * tx + ty * ty
            if distance < radius * radius:
                x = int(int(tx / 2.0) * (math.sqrt(distance) / r) + cx)
                y = int(int(ty / 2.0) * (math.sqrt(distance) / r) + cy)
                if x < w and y < h:
                    new_img[j, i, 0] = img[y, x, 0]
                    new_img[j, i, 1] = img[y, x, 1]
                    new_img[j, i, 2] = img[y, x, 2]
    return new_img

def enlarge_line_effect(img):
    """
    Apply a line enlargement effect to the input image (optimized version).

    Parameters:
    - img: OpenCV image (BGR format)

    Returns:
    - enlarged_img: Image with the applied line enlargement effect (BGR format)
    """
    h, w, _ = img.shape

    cx = w / 2
    cy = h / 2
    radius = 200
    r = int(radius / 2.0)

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Calculate distance from the center
    distance = (x - cx)**2 + (y - cy)**2

    # Create a mask for pixels within the specified radius
    mask = distance < radius**2

    # Calculate new coordinates
    new_x = (x[mask] - cx) / 2.0 * (np.sqrt(distance[mask]) / r) + cx
    new_y = (y[mask] - cy) / 2.0 * (np.sqrt(distance[mask]) / r) + cy

    # Clip coordinates to stay within image bounds
    new_x = np.clip(new_x, 0, w - 1).astype(int)
    new_y = np.clip(new_y, 0, h - 1).astype(int)

    # Create an output image with the same shape as the input
    enlarged_img = np.zeros_like(img)

    # Assign values from the original image to the new coordinates
    enlarged_img[y[mask], x[mask]] = img[new_y, new_x]
    return enlarged_img

def enlarge_line_effect_fixed(img):
    """
    Apply a line enlargement effect to the input image (fixed version).

    Parameters:
    - img: OpenCV image (BGR format)

    Returns:
    - enlarged_img: Image with the applied line enlargement effect (BGR format)
    """
    h, w, _ = img.shape

    cx = w / 2
    cy = h / 2
    radius = 200
    r = int(radius / 2.0)

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Calculate distance from the center
    distance = (x - cx)**2 + (y - cy)**2

    # Create masks for pixels within and outside the specified radius
    inside_radius_mask = distance < radius**2
    outside_radius_mask = ~inside_radius_mask

    # Calculate new coordinates for pixels within the radius
    new_x = (x[inside_radius_mask] - cx) / 2.0 * (np.sqrt(distance[inside_radius_mask]) / r) + cx
    new_y = (y[inside_radius_mask] - cy) / 2.0 * (np.sqrt(distance[inside_radius_mask]) / r) + cy

    # Clip coordinates to stay within image bounds
    new_x = np.clip(new_x, 0, w - 1).astype(int)
    new_y = np.clip(new_y, 0, h - 1).astype(int)

    # Create an output image with the same shape as the input
    enlarged_img = np.zeros_like(img)

    # Assign values from the original image to the new coordinates within the radius
    enlarged_img[y[inside_radius_mask], x[inside_radius_mask]] = img[new_y, new_x]

    # Copy values from the original image for pixels outside the radius
    enlarged_img[y[outside_radius_mask], x[outside_radius_mask]] = img[y[outside_radius_mask], x[outside_radius_mask]]
    return enlarged_img

def enlarge_effect_optimized(img):
    h, w, n = img.shape
    cx = w / 2
    cy = h / 2
    radius = 100
    r = int(radius / 2.0)
    
    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Calculate differences from the center
    tx = x - cx
    ty = y - cy
    
    # Calculate distance squared
    distance_squared = tx**2 + ty**2
    
    # Create a mask for pixels within the specified radius
    mask = distance_squared < radius**2
    
    # Calculate new coordinates
    new_x = (tx[mask] / 2.0 * (np.sqrt(distance_squared[mask]) / r) + cx).astype(int)
    new_y = (ty[mask] / 2.0 * (np.sqrt(distance_squared[mask]) / r) + cy).astype(int)
    
    # Clip coordinates to stay within image bounds
    new_x = np.clip(new_x, 0, w - 1)
    new_y = np.clip(new_y, 0, h - 1)
    
    # Create an output image with the same shape as the input
    new_img = np.zeros_like(img)
    
    # Assign values from the original image to the new coordinates
    new_img[y[mask], x[mask]] = img[new_y, new_x]
    return new_img

def reduce_effect_optimized(img):
    h, w, n = img.shape
    cx = w / 2
    cy = h / 2
    radius = 100
    r = int(radius / 2.0)
    compress = 8

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Calculate polar coordinates
    tx = x - cx
    ty = y - cy
    polar_coords = np.sqrt(np.sqrt(tx**2 + ty**2))

    # Calculate new coordinates
    new_x = (cx + (polar_coords * compress * np.cos(np.arctan2(ty, tx)))).astype(int)
    new_y = (cy + (polar_coords * compress * np.sin(np.arctan2(ty, tx)))).astype(int)

    # Clip coordinates to stay within image bounds
    new_x = np.clip(new_x, 0, w - 1)
    new_y = np.clip(new_y, 0, h - 1)

    # Index the original image with the new coordinates
    new_img = img[new_y, new_x]
    return new_img
def apply_wave_effect(img, amplitude=10, frequency=0.1):
    """
    Apply a horizontal wave effect to the input image.

    Parameters:
    - img: OpenCV image (BGR format)
    - amplitude: Amplitude of the wave effect
    - frequency: Frequency of the wave effect

    Returns:
    - waved_img: Image with the applied wave effect (BGR format)
    """
    h, w, _ = img.shape

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Apply a horizontal wave distortion
    wave = amplitude * np.sin(2 * np.pi * frequency * x / w)

    # Displace the x-coordinates by the wave effect
    new_x = (x + wave).astype(int)

    # Clip the new_x values to stay within image bounds
    new_x = np.clip(new_x, 0, w - 1)

    # Create an output image with the same shape as the input
    waved_img = np.zeros_like(img)

    # Assign values from the original image to the new coordinates
    waved_img[y, x] = img[y, new_x]
    return waved_img

if __name__ == "__main__":
    print('hello')
    cap = cv2.VideoCapture(0)
    width=640
    height=480
    cap.set(3, width)  # Set width
    cap.set(4, height)  # Set height
    start(cap)
