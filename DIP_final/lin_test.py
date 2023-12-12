#-----------------zoom in and shrink---------------------
def zoom_in(self):
        self.resizing("zoom_in")
         
def shrink(self):
        self.resizing("shrink")

def resizing(self, method):
        try:
            a = 2
            b = 2 
            width, height = self.cap.size

            # Calculate new w and h.
            if method == "zoom_in":
                new_width = int(width * a)
                new_height = int(height * b)

            elif method == "shrink":
                new_width = int(width / a)
                new_height = int(height / b)

            resized_image = self.cap.resize((new_width, new_height), Image.BILINEAR) # Bilinear interpolation
            self.cap = resized_image
           
            
            
#----------gray_level_slicing---------------------------------------------------
    
def gray_level_slicing(self):
	image=self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #image variable is the gary image 
#----------------sharpen----------------
    
def sharpen(self, sigma=100):    
    
    blur_img = cv2.GaussianBlur(self, (0, 0), sigma)
    image = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

#-----------smoothing-------------------------------------------------
    

def apply_smoothing(self):
        
            smoothing_level = 2
            image = self.image.filter(ImageFilter.GaussianBlur(smoothing_level))
            
    
    

#----------convolution---------------------------------------------------
	image=image=self.cap.read()
	# 設定kernel size為5x5
	kernel_size = 5
# 使用numpy建立 5*5且值為1/(5**2)的矩陣作為kernel。
	kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size**2
	image = cv2.filter2D(image, dst=-1, kernel=kernel, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    
#----------edge computation---------------------------------------------------
	image=self.cap.read()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 轉成灰階
	image = cv2.medianBlur(image, 7)                 # 模糊化，去除雜訊
	image = cv2.Laplacian(image, -1, 1, 5)        # 偵測邊緣
	
#----------erosion---------------------------------------------------
	kernel = cv2.getStructuringElement(shape, ksize)# 返回指定大小形狀的結構元素
# shape 的內容：cv2.MORPH_RECT ( 矩形 )、cv2.MORPH_CROSS ( 十字交叉 )、cv2.MORPH_ELLIPSE ( 橢圓形 )
# ksize 的格式：(x, y)
	

	kernel = cv2.getStructuringElement(shape, ksize)
	image=self.cap.read()
	image = cv2.erode(image, kernel)   # 侵蝕
	image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

	image = cv2.erode(image, kernel)     # 先侵蝕，將白色小圓點移除
	
#----------dilation---------------------------------------------------
	ernel = cv2.getStructuringElement(shape, ksize)# 返回指定大小形狀的結構元素
	
	kernel = cv2.getStructuringElement(shape, ksize)
	image=self.cap.read()
	image = cv2.dilate(image, kernel)  # 擴張
	image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

	image = cv2.erode(image, kernel)     # 先侵蝕
	

	image = cv2.dilate(image, kernel)    # 再膨脹
	
	
#----------btightness---------------------------------------------------
	)
	#from PIL import Image, ImageEnhance               camera.py need to import this function

	image=self.cap.read()
	brightness = ImageEnhance.Brightness(image)  # 設定 img 要加強亮度
	image = brightness.enhance(1.5)           # 提高亮度
	
	
#----------gray_level_slicing---------------------------------------------------
	image=self.cap.read()
	brightness = ImageEnhance.Brightness(image)  # 設定 img 要加強亮度
	image = brightness.enhance(0.5)           # 降低亮度

























    
    
    
    
    
    
    
    
    
    
    