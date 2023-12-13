import cv2
import numpy as np
 
 
def oilPainting(img, templateSize, bucketSize, step):#templateSize模板大小,bucketSize桶阵列,step模板滑动步长
 
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
    return  oilImg
    
 
img = cv2.imread(r'/Users/liushiwen/Desktop/大四上/數位影像處理/DIP_homework/DIP_final/powerman.jpg', cv2.IMREAD_ANYCOLOR)
oil = oilPainting(img,4,8,2)
cv2.imshow('youhua',oil)
cv2.imwrite(r'/Users/liushiwen/Desktop/大四上/數位影像處理/DIP_homework/DIP_final/powerman_oil.jpg',oil)
cv2.waitKey(0)
cv2.destroyAllWindows()




        # self.reduce_effect_optimized_btn = self.create_button("Reduce Effect Optimized", lambda: reduce_effect_optimized(self.img))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.apply_mosaic_effect_btn = self.create_button("Apply Mosaic Effect", lambda: apply_mosaic_effect(self.img))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)      

        # self.enlarge_line_effect_btn = self.create_button("Enlarge Line Effect", lambda: enlarge_line_effect(self.img))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.grayscale_conversion_btn = self.create_button("Grayscale Conversion", lambda: grayscale_conversion(self.img))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)    

        # self.sepia_tone_btn = self.create_button("Sepia Tone", lambda: sepia_tone(self.img))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.invert_colors_btn = self.create_button("Invert Colors", lambda: invert_colors(self.img))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)    

        # self.posterize_effect_btn = self.create_button("Posterize Effect", lambda: posterize_effect(self.img))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.solarize_effect_btn = self.create_button("Solarize Effect", lambda: solarize_effect(self.img, 128))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)   

        # self.bitwise_not_effect_btn = self.create_button("Bitwise Not Effect", lambda: bitwise_not_effect(self.img))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.emboss_effect_btn = self.create_button("Emboss Effect", lambda: emboss_effect(self.img))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)   

        # self.blur_effect_btn = self.create_button("Blur Effect", lambda: blur_effect(self.img, 3))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.sharpen_effect_btn = self.create_button("Sharpen Effect", lambda: sharpen_effect(self.img))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)   

        # self.apply_oil_painting_btn = self.create_button("Apply Oil Painting", lambda: apply_oil_painting(self.img))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.apply_sketch_effect_btn = self.create_button("Apply Sketch Effect", lambda: apply_sketch_effect(self.img))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)   

        # self.apply_watercolor_effect_btn = self.create_button("Apply Watercolor Effect", lambda: apply_watercolor_effect(self.img))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.apply_dreamy_effect_btn = self.create_button("Apply Dreamy Effect", lambda: apply_dreamy_effect(self.img))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)     

        # self.apply_pencil_sketch_btn = self.create_button("Apply Pencil Sketch", lambda: apply_pencil_sketch(self.img))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.apply_pixelate_btn = self.create_button("Apply Pixelate", lambda: apply_pixelate(self.img))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)     

        # self.apply_cartoonize_btn = self.create_button("Apply Cartoonize", lambda: apply_cartoonize(self.img))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.apply_gaussian_blur_btn = self.create_button("Apply Gaussian Blur", lambda: apply_gaussian_blur(self.img))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)     
           
        # self.apply_halftone_btn = self.create_button("Apply Halftone", lambda: apply_halftone(self.img, 1))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.apply_color_splash_btn = self.create_button("Apply Color Splash", lambda: apply_color_splash(self.img))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)     

        # self.apply_vignette_btn = self.create_button("Apply Vignette", lambda: apply_vignette(self.img, 0.1))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.apply_gradient_map_btn = self.create_button("Apply Gradient Map", lambda: apply_gradient_map(self.img))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10) 

        # self.apply_lens_flare_btn = self.create_button("Apply Lens Flare", lambda: apply_lens_flare(self.img))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.apply_double_exposure_btn = self.create_button("Apply Double Exposure", lambda: apply_double_exposure(self.img))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)  

        # self.apply_kaleidoscope_btn = self.create_button("Apply Kaleidoscope", lambda: apply_kaleidoscope(self.img, 10))
        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.apply_pixelate_btn, enlarge_effect), bg="white")
        self.apply_pixelate_btn.grid(row=0, column=2, padx=10)

        # self.apply_glitch_art_btn = self.create_button("Apply Glitch Art", lambda: apply_glitch_art(self.img, 0.01))
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)