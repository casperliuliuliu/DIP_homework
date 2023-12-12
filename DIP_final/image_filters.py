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
    # img = enlarge_effect_fixed(img)
    # img = reduce_effect_optimized(img)
    # img = apply_mosaic_effect(img)
    # img = low_quality_effect(img, 10)
    # img = enlarge_line_effect(img) # still got problems.
    # img = grayscale_conversion(img)
    img = sepia_tone(img) # This is boring.
    # img = invert_colors(img)

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

def enlarge_line_effect(img):
    h, w, _ = img.shape

    cx = w / 2
    cy = h / 2
    radius = 200
    r = int(radius / 2.0)

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Calculate distance from the center
    distance = np.abs(y - cy)

    # Create a mask for pixels within the specified distance
    mask = distance < radius

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

def enlarge_effect_fixed(img, radius = 100):
    h, w, _ = img.shape

    cx = w / 2
    cy = h / 2
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

def oilPainting(img, templateSize=4, bucketSize=8, step=2):#templateSize模板大小,bucketSize桶阵列,step模板滑动步长
 
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = ((gray/256)*bucketSize).astype(int)                          #灰度图在桶中的所属分区
    h,w = img.shape[:2]
     
    oilImg = np.zeros(img.shape, np.uint8)                              #用来存放过滤图像
     
    for i in range(0,h,step):
        
        top = i-templateSize
        bottom = i+templateSize+1
        if top < 0:
            top = 0
        if bottom >= h:
            bottom = h-1
            
        for j in range(0,w,step):
            
            left = j-templateSize
            right = j+templateSize+1
            if left < 0:
                left = 0
            if right >= w:
                right = w-1
                
            # 灰度等级统计
            buckets = np.zeros(bucketSize,np.uint8)                     #桶阵列，统计在各个桶中的灰度个数
            bucketsMean = [0,0,0]                                       #对像素最多的桶，求其桶中所有像素的三通道颜色均值
            #对模板进行遍历
            for c in range(top,bottom):
                for r in range(left,right):
                    buckets[gray[c,r]] += 1                         #模板内的像素依次投入到相应的桶中，有点像灰度直方图
    
            maxBucket = np.max(buckets)                                 #找出像素最多的桶以及它的索引
            maxBucketIndex = np.argmax(buckets)
            
            for c in range(top,bottom):
                for r in range(left,right):
                    if gray[c,r] == maxBucketIndex:
                        bucketsMean += img[c,r]
            bucketsMean = (bucketsMean/maxBucket).astype(int)           #三通道颜色均值
            
            # 油画图
            for m in range(step):
                for n in range(step):
                    oilImg[m+i,n+j] = (bucketsMean[0],bucketsMean[1],bucketsMean[2])
    return oilImg

def low_quality_effect(img, resolution = 10):
    block_size = resolution
    h, w, _ = img.shape
    mosaic = cv2.resize(img, (int(w/block_size),int(h/block_size)), interpolation=cv2.INTER_LINEAR)   # 根據縮小尺寸縮小
    mosaic = cv2.resize(mosaic, (w,h), interpolation=cv2.INTER_NEAREST) 
    return mosaic
    
def apply_mosaic_effect(img, block_size=10):
    h, w, _ = img.shape

    # Calculate the number of mosaic blocks in each dimension
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size

    # Create an output image with the same shape as the input
    mosaic_img = np.zeros_like(img)

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Calculate the region for each mosaic block
            start_h = i * block_size
            end_h = (i + 1) * block_size
            start_w = j * block_size
            end_w = (j + 1) * block_size

            # Calculate the average color of the region
            avg_color = np.mean(img[start_h:end_h, start_w:end_w], axis=(0, 1))

            # Fill the mosaic block with the average color
            mosaic_img[start_h:end_h, start_w:end_w] = avg_color

    return mosaic_img

def reduce_effect_optimized(img, radius=100):
    h, w, n = img.shape
    cx = w / 2
    cy = h / 2
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

def grayscale_conversion(img):
    """
    Convert the input image to grayscale.

    Parameters:
    - img: OpenCV image (BGR format)

    Returns:
    - grayscale_img: Grayscale image (single channel)
    """
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayscale_img

def sepia_tone(img):
    """
    Apply a sepia tone filter to the input image.

    Parameters:
    - img: OpenCV image (BGR format)

    Returns:
    - sepia_img: Image with sepia tone applied (BGR format)
    """
    # Define the sepia tone matrix
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])

    # Apply the sepia tone transformation
    sepia_img = cv2.transform(img, sepia_matrix)

    # Clip values to stay within the valid color range
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)

    return sepia_img

def invert_colors(img):
    """
    Invert the colors of the input image.

    Parameters:
    - img: OpenCV image (BGR format)

    Returns:
    - inverted_img: Image with inverted colors (BGR format)
    """
    inverted_img = cv2.bitwise_not(img)
    return inverted_img

if __name__ == "__main__":
    print('hello')
    cap = cv2.VideoCapture(0)
    width=640
    height=480
    cap.set(3, width)  # Set width
    cap.set(4, height)  # Set height

    start(cap)
