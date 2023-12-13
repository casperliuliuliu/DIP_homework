import cv2
from PIL import Image, ImageTk
import numpy as np
import math
from sklearn.cluster import MiniBatchKMeans
import colorsys
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



def apply_zoom_in(img, a=1.2, b=1.2):
    """
    Zoom in on the input image.

    Parameters:
    - img: OpenCV image (BGR format)
    - a: Zoom factor for the width (horizontal scaling)
    - b: Zoom factor for the height (vertical scaling)

    Returns:
    - processed_img: Image with zoom effect applied (BGR format)
    """
    # Get the original image size
    original_height, original_width, _ = img.shape

    # Resize the cropped image back to the original size
    processed_img = cv2.resize(img, (int(original_width*a), int(original_height*b)))
    return processed_img

def apply_zoom_out(img, a = 2, b = 2):
    """
    Zoom out on the input image.

    Parameters:
    - img: OpenCV image (BGR format)
    - a: Zoom factor for the width (horizontal scaling)
    - b: Zoom factor for the height (vertical scaling)

    Returns:
    - processed_img: Image with zoom-out effect applied (BGR format)
    """
    # Get the original image size
    original_height, original_width, _ = img.shape

    # Calculate the region to crop (1/2 of the original size)
    new_width = original_width // a  # Crop from 1/4 to 3/4 of the width
    new_height = original_height // b  # Crop from 1/4 to 3/4 of the height

    # Resize the cropped image back to the original size
    processed_img = cv2.resize(img, (new_width, new_height))
    return processed_img

def apply_flip_right_to_left(img):
    """
    Flip the input image horizontally (right to left).

    Parameters:
    - img: OpenCV image (BGR format)

    Returns:
    - flipped_img: Image with horizontal flip applied (BGR format)
    """
    flipped_img = cv2.flip(img, 1)  # 1 denotes horizontal flip
    return flipped_img

def apply_flip_top_to_down(img):
    """
    Flip the input image vertically (top to down).

    Parameters:
    - img: OpenCV image (BGR format)

    Returns:
    - flipped_img: Image with vertical flip applied (BGR format)
    """
    flipped_img = cv2.flip(img, 0)  # 1 denotes horizontal flip
    return flipped_img

def apply_show_image_difference(current_img):
    """
    Show the absolute difference between the current image and the last image.

    Parameters:
    - current_img: OpenCV image (BGR format) representing the current frame

    Returns:
    - diff_img: Image showing the absolute difference between the current and last images (BGR format)
    """
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

def apply_enlarge_line_effect(img):
    """
    Apply a line enlargement effect to the input image.

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

def apply_enlarge_effect_fixed(img, radius = 100):
    """
    Apply a fixed-radius enlargement effect to the input image.

    Parameters:
    - img: OpenCV image (BGR format)
    - radius: Radius of the enlargement effect

    Returns:
    - enlarged_img: Image with the applied fixed-radius enlargement effect (BGR format)
    """
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

def apply_oil_painting(img, templateSize=4, bucketSize=8, step=2):#templateSize模板大小,bucketSize桶阵列,step模板滑动步长
 
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

def apply_colorOilPainting(img, templateSize=4, bucketSize=8, step=2):
    # Convert the image to LAB color space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Extract the L channel
    l_channel = lab_img[:, :, 0]

    # Quantize the L channel
    quantized_l = ((l_channel / 256) * bucketSize).astype(int)

    h, w = img.shape[:2]
    oilImg = np.zeros_like(img)

    for i in range(0, h, step):
        top, bottom = max(0, i - templateSize), min(h, i + templateSize + 1)

        for j in range(0, w, step):
            left, right = max(0, j - templateSize), min(w, j + templateSize + 1)

            # Use numpy for efficient histogram calculation
            buckets, bin_edges = np.histogram(quantized_l[top:bottom, left:right].ravel(),
                                             bins=bucketSize, range=(0, bucketSize))
            maxBucketIndex = np.argmax(buckets)

            # Use boolean indexing for efficient mean calculation
            pixels_in_max_bucket = (quantized_l[top:bottom, left:right] == maxBucketIndex)
            buckets_mean = np.mean(img[top:bottom, left:right][pixels_in_max_bucket], axis=(0, 1)).astype(int)

            # Assign the mean color to the corresponding region in oilImg
            oilImg[i:i + step, j:j + step] = buckets_mean

    return oilImg


def apply_new_colorOilPainting(img, templateSize=4, bucketSize=8, step=2):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_quantized = (gray * bucketSize // 256).astype(int)

    h, w = img.shape[:2]
    oil_img = np.zeros_like(img, dtype=np.uint8)

    for i in range(0, h, step):
        top, bottom = max(0, i - templateSize), min(h, i + templateSize + 1)

        for j in range(0, w, step):
            left, right = max(0, j - templateSize), min(w, j + templateSize + 1)

            # Use numpy for efficient histogram calculation
            buckets, _ = np.histogram(gray_quantized[top:bottom, left:right].ravel(),
                                      bins=bucketSize, range=(0, bucketSize))
            max_bucket_index = np.argmax(buckets)

            # Use boolean indexing for efficient mean calculation
            pixels_in_max_bucket = (gray_quantized[top:bottom, left:right] == max_bucket_index)
            buckets_mean = np.mean(img[top:bottom, left:right][pixels_in_max_bucket], axis=(0, 1)).astype(int)

            # Assign the mean color to the corresponding region in oil_img
            oil_img[i:i + step, j:j + step] = buckets_mean

    return oil_img
    
def apply_mosaic_effect(img, block_size=10):
    """
    Apply a mosaic effect to the input image.

    Parameters:
    - img: OpenCV image (BGR format)
    - block_size: Size of the mosaic blocks

    Returns:
    - mosaic_img: Image with the applied mosaic effect (BGR format)
    """
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

def apply_reduce_effect_optimized(img, radius=100):
    """
    Apply a radial reduction effect to the input image in an optimized way.

    Parameters:
    - img: OpenCV image (BGR format)
    - radius: Radius of the reduction effect

    Returns:
    - new_img: Image with the applied radial reduction effect (BGR format)
    """
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

def apply_grayscale_conversion(img):
    """
    Convert the input image to grayscale.

    Parameters:
    - img: OpenCV image (BGR format)

    Returns:
    - grayscale_img: Grayscale image (single channel)
    """
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayscale_img

def apply_sepia_tone(img):
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

def apply_invert_colors(img):
    """
    Invert the colors of the input image.

    Parameters:
    - img: OpenCV image (BGR format)

    Returns:
    - inverted_img: Image with inverted colors (BGR format)
    """
    inverted_img = cv2.bitwise_not(img)
    return inverted_img
def apply_posterize_effect(img, num_colors=16):
    """
    Reduce the number of colors in the image to create a poster-like effect.

    Parameters:
    - img: OpenCV image (BGR format)
    - num_colors: Number of colors to retain in the posterized image

    Returns:
    - posterized_img: Image with the applied posterize effect (BGR format)
    """
    # Reshape the image to a 2D array of pixels
    pixels = img.reshape((-1, 3))

    # Convert to float32 for k-means clustering
    pixels = np.float32(pixels)

    # Use MiniBatchKMeans for faster clustering
    kmeans = MiniBatchKMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Assign labels to the centers and convert back to 8-bit values
    labels = kmeans.predict(pixels)
    centers = np.uint8(kmeans.cluster_centers_)

    # Map the labels to centers and reshape back to the original image
    segmented_img = centers[labels]
    posterized_img = segmented_img.reshape(img.shape)

    return posterized_img

def apply_bitwise_not_effect(img):
    """
    Invert some of the pixel values to create a solarization effect.

    Parameters:
    - img: OpenCV image (BGR format)
    - threshold: Threshold value for pixel intensity inversion

    Returns:
    - solarized_img: Image with the applied solarization effect (BGR format)
    """
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply solarization effect by inverting pixel values above the threshold
    bitnot_img = cv2.bitwise_not(gray_img)

    # Convert the grayscale solarized image back to BGR color space
    bitnot_img = cv2.cvtColor(bitnot_img, cv2.COLOR_GRAY2BGR)

    return bitnot_img

def apply_solarize_effect(img, threshold=128):
    """
    Invert some of the pixel values to create a solarization effect.

    Parameters:
    - img: OpenCV image (BGR format)
    - threshold: Threshold value for pixel intensity inversion

    Returns:
    - solarized_img: Image with the applied solarization effect (BGR format)
    """
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply solarization effect by inverting pixel values above the threshold
    solarized_img = cv2.bitwise_not(gray_img, mask=(gray_img > threshold).astype(np.uint8))

    # Convert the grayscale solarized image back to BGR color space
    solarized_img = cv2.cvtColor(solarized_img, cv2.COLOR_GRAY2BGR)

    return solarized_img

def apply_emboss_effect(img):
    """
    Highlight the edges of objects to give a 3D appearance.

    Parameters:
    - img: OpenCV image (BGR format)

    Returns:
    - embossed_img: Image with the applied emboss effect (BGR format)
    """
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a convolution kernel for embossing
    kernel = np.array([[0, -1, -1],
                       [1,  0, -1],
                       [1,  1,  0]])
    embossed_img = cv2.filter2D(gray_img, cv2.CV_8U, kernel)

    # Convert the grayscale embossed image back to BGR color space
    embossed_img = cv2.cvtColor(embossed_img, cv2.COLOR_GRAY2BGR)

    return embossed_img

def apply_blur_effect(img, kernel_size=5):
    """
    Apply a blur effect to the image to smooth out details.

    Parameters:
    - img: OpenCV image (BGR format)
    - kernel_size: Size of the blur kernel

    Returns:
    - blurred_img: Image with the applied blur effect (BGR format)
    """
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blurred_img

def apply_sharpen_effect(img):
    """
    Enhance the edges of objects to make them more defined.

    Parameters:
    - img: OpenCV image (BGR format)

    Returns:
    - sharpened_img: Image with the applied sharpening effect (BGR format)
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    # kernel = np.array([[ 0, -1,  0],
    #                    [-1,  5, -1],
    #                    [ 0, -1,  0]])
    sharpened_img = cv2.filter2D(img, cv2.CV_8U, kernel)
    return sharpened_img

def apply_watercolor_effect(img, sigma_s=60, sigma_r=0.4):
    # 加载图像
    image = img
    # 应用水彩效果
    watercolor = cv2.stylization(image, sigma_s=sigma_s, sigma_r=sigma_r)
    return watercolor

def apply_sketch_effect(img):
    # 加载图像
    image = img
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 将灰度图像转换为边缘图像
    edges = cv2.Canny(gray_image, 30, 100)
    # 将边缘图像转换为彩色图像
    sketch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return sketch


def apply_dreamy_effect(img, blend_alpha=0.5):
    # 加载图像
    image = img
    # 将图像转换为浮点数格式
    image = image.astype(float) / 255.0
    # 应用高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    # 将图像与模糊图像混合
    dreamy = cv2.addWeighted(image, 1 - blend_alpha, blurred, blend_alpha, 0)
    return dreamy


def apply_pencil_sketch(img, ksize=5):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the grayscale image
    blurred_img = cv2.GaussianBlur(gray_img, (ksize, ksize), 0)

    # Invert the blurred image to create a pencil sketch effect
    sketch_img = cv2.divide(gray_img, 255 - blurred_img, scale=256.0)

    return cv2.cvtColor(cv2.merge([sketch_img, sketch_img, sketch_img]), cv2.COLOR_BGR2RGB)

def apply_pixelate(img, block_size=10):
    h, w = img.shape[:2]

    # Resize the image to a small size
    small_img = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)

    # Resize the small image back to the original size
    pixelated_img = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_NEAREST)

    return pixelated_img

def apply_cartoonize(img, num_down=2, num_bilateral=7):
    """
    Transform the image into a cartoon-style representation.

    Parameters:
    - img: Input image (BGR format)
    - num_down: Number of downsampled levels
    - num_bilateral: Number of bilateral filtering steps

    Returns:
    - cartoon_img: Cartoon-style image (BGR format)
    """
    # Downsample the image multiple times for faster processing
    img_color = img
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)

    # Apply bilateral filter to reduce noise and smooth the image
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

    # Upsample the image back to its original size
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median blur to the grayscale image
    img_blur = cv2.medianBlur(img_gray, 7)

    # Create an edge mask using adaptive thresholding
    edges = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

    # Combine the color and edge images to create the cartoon effect
    cartoon_img = cv2.bitwise_and(img_color, img_color, mask=edges)

    return cartoon_img


def apply_gaussian_blur(img, sigma=1.0):
    """
    Apply a blur with a Gaussian filter to create a smoother appearance.

    Parameters:
    - img: Input image (BGR format)
    - sigma: Standard deviation of the Gaussian filter

    Returns:
    - blurred_img: Blurred image (BGR format)
    """
    # Apply Gaussian blur to the image
    blurred_img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    return blurred_img


def apply_halftone(img, radius=3):
    """
    Simulate the appearance of halftone printing.

    Parameters:
    - img: Input image (BGR format)
    - radius: Radius for the halftone effect

    Returns:
    - halftone_img: Halftone-style image (BGR format)
    """
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to create a binary halftone pattern
    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

    # Create a circular pattern for halftone using morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius, 2 * radius))
    halftone_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Convert the binary halftone pattern to color format
    halftone_img = cv2.cvtColor(halftone_img, cv2.COLOR_GRAY2BGR)

    return halftone_img

def apply_color_splash(img, color=(0, 0, 0)):
    """
    Convert the image to grayscale and highlight specific colored regions.

    Parameters:
    - img: Input image (BGR format)
    - color: Color for highlighting specified regions (BGR format)

    Returns:
    - splash_img: Image with color splash effect (BGR format)
    """
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a mask for non-black pixels in the original image
    mask = gray_img > 128

    # Create a copy of the original image and apply color splash
    splash_img = img.copy()
    splash_img[mask] = color

    return splash_img


def apply_vignette(img, strength=0.7):
    """
    Darken the corners of the image to draw attention to the center.

    Parameters:
    - img: Input image (BGR format)
    - strength: Strength of the vignette effect

    Returns:
    - vignette_img: Image with vignette effect (BGR format)
    """
    # Create a circular mask for the vignette effect
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * strength), 255, -1)

    # Apply the mask to darken the corners
    vignette_img = cv2.bitwise_and(img, img, mask=mask)

    return vignette_img


def apply_gradient_map(img, colors=None):
    """
    Apply a color gradient based on pixel intensity.

    Parameters:
    - img: Input image (BGR format)
    - colors: List of colors for the gradient, default is None (uses rainbow colors)

    Returns:
    - gradient_img: Image with gradient map effect (BGR format)
    """
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a default rainbow color map if colors are not provided
    if colors is None:
        colors = [tuple(int(255 * x) for x in colorsys.hsv_to_rgb(i / 360.0, 1.0, 1.0))
                  for i in range(360)]

    # Normalize pixel intensities to the range [0, 1]
    normalized_intensity = gray_img / 255.0

    # Map pixel intensities to colors based on the gradient
    gradient_img = np.zeros_like(img)
    for color, intensity in zip(colors, normalized_intensity.flatten()):
        gradient_img += (img * intensity).astype(np.uint8)

    return gradient_img

def apply_lens_flare(img, flare_intensity=0.5, num_flare_circles=6):
    """
    Simulate lens flares to add a creative touch.

    Parameters:
    - img: Input image (BGR format)
    - flare_intensity: Intensity of the lens flares, default is 0.5
    - num_flare_circles: Number of flare circles to add, default is 6

    Returns:
    - flared_img: Image with lens flare effect (BGR format)
    """
    h, w, _ = img.shape
    flared_img = img.copy()

    # Generate random lens flare circles
    for _ in range(num_flare_circles):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        flare_color = np.random.rand(3) * 255 * flare_intensity
        cv2.circle(flared_img, (x, y), np.random.randint(10, 100), flare_color, -1)

    return flared_img


def apply_double_exposure(img1, alpha=0.5):
    """
    Combine two images to create a double exposure effect.

    Parameters:
    - img1: First input image (BGR format)
    - img2: Second input image (BGR format)
    - alpha: Blending strength, where 0.0 means only img1, and 1.0 means only img2, default is 0.5

    Returns:
    - double_exposure_img: Image with double exposure effect (BGR format)
    """
    global last_img
    if isinstance(last_img, int):
        last_img = img1

        return img1
    
    double_exposure_img = cv2.addWeighted(img1, alpha, last_img, 1 - alpha, 0)
    last_img = img1

    return double_exposure_img

def apply_kaleidoscope(img, num_sections=6):
    """
    Replicate and mirror portions of the image to create a kaleidoscopic effect.

    Parameters:
    - img: Input image (BGR format)
    - num_sections: Number of kaleidoscope sections, default is 6

    Returns:
    - kaleidoscope_img: Image with kaleidoscope effect (BGR format)
    """
    h, w, _ = img.shape
    kaleidoscope_img = np.zeros_like(img)

    # Split the image into sections and replicate/mirror them
    section_width = w // num_sections
    for i in range(num_sections):
        section = img[:, i * section_width: (i + 1) * section_width, :]
        kaleidoscope_img[:, i * section_width: (i + 1) * section_width, :] = cv2.flip(section, 1)

    return kaleidoscope_img


def apply_glitch_art(img, intensity=0.1):
    """
    Introduce digital glitches and artifacts for a futuristic and surreal look.

    Parameters:
    - img: Input image (BGR format)
    - intensity: Intensity of the glitch effect, default is 0.1

    Returns:
    - glitched_img: Image with glitch art effect (BGR format)
    """
    h, w, _ = img.shape
    glitched_img = img.copy()

    # Introduce random glitches based on intensity using vectorized operations
    num_glitches = int(intensity * h * w)
    
    # Generate random indices for glitches
    y_indices = np.random.randint(0, h, num_glitches)
    x_indices = np.random.randint(0, w, num_glitches)

    # Generate random colors for glitches
    glitch_colors = np.random.randint(0, 255, size=(num_glitches, 3))

    # Apply glitches to the image
    glitched_img[y_indices, x_indices, :] = glitch_colors

    return glitched_img



def apply_fractal_art_optimized(img, num_iterations=5):
    """
    Generate fractal patterns within the image for a complex and intricate appearance.

    Parameters:
    - img: Input image (BGR format)
    - num_iterations: Number of fractal iterations, default is 5

    Returns:
    - fractal_img: Image with fractal art effect (BGR format)
    """
    h, w, _ = img.shape
    fractal_img = img.copy()

    # Generate fractal patterns within the image
    for _ in range(num_iterations):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        size = np.random.randint(10, 50)
        color = tuple(np.random.randint(0, 255, size=(3,)).tolist())
        cv2.circle(fractal_img, (x, y), size, color, -1)

    return fractal_img

def transform(img):
    """
    Test the function to do the filter.

    Parameters:
    - img: OpenCV image (BGR format)

    Returns:
    - img: Image with testing filter (BGR format)
    """
    global last_img
    # img = apply_zoom_in(img, 1, 2)
    # img = apply_zoom_out(img, 2, 4)
    # img = apply_flip_right_to_left(img)
    # img = apply_flip_top_to_down(img)
    # img = apply_show_image_difference(img)
    # img = apply_enlarge_effect_fixed(img)
    # img = apply_reduce_effect_optimized(img)
    # img = apply_mosaic_effect(img)
    # img = apply_enlarge_line_effect(img) # still got problems.
    # img = apply_grayscale_conversion(img)
    # img = apply_sepia_tone(img) # This is boring.
    # img = apply_invert_colors(img)
    # img = apply_posterize_effect(img)
    # img = apply_solarize_effect(img, 128)
    # img = apply_bitwise_not_effect(img)
    # img = apply_emboss_effect(img)
    # img = apply_blur_effect(img, 3)
    # img = apply_sharpen_effect(img)
    # img = apply_oil_painting(img) # running too slow, need optimization
    # img = apply_sketch_effect(img)
    # img = apply_watercolor_effect(img)
    img = apply_dreamy_effect(img)
    # img = apply_pencil_sketch(img)
    # img = apply_pixelate(img)
    # img = apply_cartoonize(img)
    # img = apply_gaussian_blur(img)
    # img = apply_halftone(img, 1)
    # img = apply_color_splash(img)
    # img = apply_vignette(img, 0.1)
    # img = apply_gradient_map(img)
    # img = apply_lens_flare(img)
    # img = apply_double_exposure(img)
    # img = apply_kaleidoscope(img,10)
    # img = apply_glitch_art(img, 0.01)
    # img = apply_fractal_art_optimized(img,1 ) # this might be repeated
    return img
if __name__ == "__main__":
    print('hello')
    cap = cv2.VideoCapture(0)
    width=640
    height=480
    cap.set(3, width)  # Set width
    cap.set(4, height)  # Set height

    start(cap)
