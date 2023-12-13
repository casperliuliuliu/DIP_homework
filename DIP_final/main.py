import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime
import tkmacosx
from funcs import change_color
from image_filters import *
class VideoApp:
    option = 'ori'
    recording = False
    filename = None

    def __init__(self, root, video_source=0, width=640, height=480):
        self.root = root
        self.root.title("Video Filter App")

        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(3, width)  # Set width
        self.cap.set(4, height)  # Set height

        self.video_frame = tk.Frame(root)
        self.video_frame.pack(pady=10)

        self.canvas_left = tk.Canvas(self.video_frame)
        self.canvas_left.pack(side=tk.LEFT, padx=10)

        self.canvas_right = tk.Canvas(self.video_frame)
        self.canvas_right.pack(side=tk.LEFT, padx=10)

        self.btn_frame = ttk.Notebook(root)
        self.btn_frame.pack(pady=10)

        self.tab1_frame = ttk.Frame(self.btn_frame)
        self.tab1_functions = Func1(self.tab1_frame, self)

        # Create a frame for the second tab
        self.tab2_frame = ttk.Frame(self.btn_frame)
        self.tab2_functions = Func2(self.tab2_frame, self)

        # Add the frames as tabs to the notebook
        self.btn_frame.add(self.tab1_frame, text="BASIC")
        self.btn_frame.add(self.tab2_frame, text="ADVANCED")
        initialize_global_vars()
        self.video_update()

    def video_update(self):
        ret, frame = self.cap.read()

        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_left = Image.fromarray(img)
            img_left = ImageTk.PhotoImage(img_left)

            img_right = self.cool_filter(img)
            img_right = ImageTk.PhotoImage(img_right)

            # Update left canvas with the resized left image
            self.canvas_left.config(width=img_left.width(), height=img_left.height())
            self.canvas_left.create_image(0, 0, anchor=tk.NW, image=img_left)
            self.canvas_left.image = img_left  # Save reference to prevent garbage collection

            # Update right canvas with the resized right image
            self.canvas_right.config(width=img_right.width(), height=img_right.height())
            self.canvas_right.create_image(0, 0, anchor=tk.NW, image=img_right)
            self.canvas_right.image = img_right  # Save reference to prevent garbage collection

    def cool_filter(self, img):
        temp_img = img.copy()
    
        if red_var.get():
            temp_img[:, :, 2] = 0  # Set blue channel to 0
            temp_img[:, :, 1] = 0  # Set green channel to 0
        if blue_var.get():
            temp_img[:, :, 0] = 0  # Set red channel to 0
            temp_img[:, :, 1] = 0  # Set green channel to 0
        if green_var.get():
            temp_img[:, :, 0] = 0  # Set red channel to 0
            temp_img[:, :, 2] = 0  # Set blue channel to 0

        if zoom_in.get():
            temp_img = apply_zoom_in(temp_img)
        if zoom_out.get():
            temp_img = apply_zoom_out(temp_img)

        if flip_rl.get():
            temp_img = apply_flip_right_to_left(temp_img)
        if flip_td.get():
            temp_img = apply_flip_top_to_down(temp_img)
            
        if image_difference.get():
            temp_img = apply_show_image_difference(temp_img)
        if enlarge_effect.get():
            temp_img = apply_enlarge_effect_fixed(temp_img)
        if reduce_effect.get():
            temp_img = apply_reduce_effect_optimized(temp_img)
        # enlarge_line_effect = tk.IntVar(value=0) # still got problems.
        if mosaic_effect.get():
            temp_img = apply_mosaic_effect(temp_img)
        if grayscale_conversion.get():
            temp_img = apply_grayscale_conversion(temp_img)
        if sepia_tone.get():
            temp_img = apply_sepia_tone(temp_img)
        if invert_colors.get():
            temp_img = apply_invert_colors(temp_img)
        if posterize_effect.get():
            temp_img = apply_posterize_effect(temp_img)
        if solarize_effect.get():
            temp_img = apply_solarize_effect(temp_img)
        if bitwise_not_effect.get():
            temp_img = apply_bitwise_not_effect(temp_img)
        if emboss_effect.get():
            temp_img = apply_emboss_effect(temp_img)
        if blur_effect.get():
            temp_img = apply_blur_effect(temp_img)
        if sharpen_effect.get():
            temp_img = apply_sharpen_effect(temp_img)
        if oil_painting.get(): # running too slow, need optimization
            temp_img = apply_oil_painting(temp_img)
        if sketch_effect.get():
            temp_img = apply_sketch_effect(temp_img)
        if watercolor_effect.get():
            temp_img = apply_watercolor_effect(temp_img)
        if dreamy_effect.get():
            temp_img = apply_gaussian_blur(temp_img)
        if pencil_sketch.get():
            temp_img = apply_pencil_sketch(temp_img)
        if pixelate.get():
            temp_img = apply_pixelate(temp_img)
        if cartoonize.get():
            temp_img = apply_cartoonize(temp_img)
        if gaussian_blur.get():
            temp_img = apply_gaussian_blur(temp_img)
        if halftone.get():
            temp_img = apply_halftone(temp_img)
        if color_splash.get():
            temp_img = apply_color_splash(temp_img)
        if vignette.get():
            temp_img = apply_vignette(temp_img)
        if gradient_map.get():
            temp_img = apply_gradient_map(temp_img)
        if lens_flare.get():
            temp_img = apply_lens_flare(temp_img)
        if double_exposure.get():
            temp_img = apply_double_exposure(temp_img)
        if kaleidoscope.get():
            temp_img = apply_kaleidoscope(temp_img)
        if glitch_art.get():
            temp_img = apply_glitch_art(temp_img)
        # Call the update method after 10 milliseconds
        self.root.after(10, self.video_update)

        img_right = Image.fromarray(temp_img)
        return img_right

class Func1:
    def __init__(self, btn_frame, main_app):
        self.btn_frame = btn_frame
        self.main_app = main_app
        self.create_tab1_widgets()  # Create buttons and labels for Tab 1

    def create_tab1_widgets(self):

        self.ori_btn = ttk.Button(self.btn_frame, text="CLEAR", command=self.recover)
        self.ori_btn.grid(row=0, column=0, padx=10)

        self.red_btn = tkmacosx.Button(self.btn_frame, text="Red", command=lambda: change_color(self.red_btn, red_var), bg="white")
        self.red_btn.grid(row=0, column=1, padx=10)

        self.blue_btn = tkmacosx.Button(self.btn_frame, text="Blue", command=lambda: change_color(self.blue_btn, blue_var), bg="white")
        self.blue_btn.grid(row=0, column=2, padx=10)

        self.green_btn = tkmacosx.Button(self.btn_frame, text="Green", command=lambda: change_color(self.green_btn, green_var), bg="white")
        self.green_btn.grid(row=0, column=3, padx=10)

        self.zoom_in_btn = tkmacosx.Button(self.btn_frame, text="Zoom In", command=lambda: change_color(self.zoom_in_btn, zoom_in), bg="white")
        self.zoom_in_btn.grid(row=1, column=0, padx=10)

        self.zoom_out_btn = tkmacosx.Button(self.btn_frame, text="Zoom Out", command=lambda: change_color(self.zoom_out_btn, zoom_out), bg="white")
        self.zoom_out_btn.grid(row=1, column=1, padx=10)

        self.flip_right_to_left_btn = tkmacosx.Button(self.btn_frame, text="Flip RL", command=lambda: change_color(self.flip_right_to_left_btn, flip_rl), bg="white")
        self.flip_right_to_left_btn.grid(row=1, column=2, padx=10)

        self.flip_top_to_down_btn = tkmacosx.Button(self.btn_frame, text="Flip TD", command=lambda: change_color(self.flip_top_to_down_btn, flip_td), bg="white")
        self.flip_top_to_down_btn.grid(row=1, column=3, padx=10)

    def recover(self):
        red_var.set(0)
        self.red_btn['bg'] = "white"
        blue_var.set(0)
        self.blue_btn['bg'] = "white"
        green_var.set(0)
        self.green_btn['bg'] = "white"
        zoom_in.set(0)
        self.zoom_in_btn['bg'] = "white"
        zoom_out.set(0)
        self.zoom_out_btn['bg'] = "white"
        flip_rl.set(0)
        self.flip_right_to_left_btn['bg'] = "white"
        flip_td.set(0)
        self.flip_top_to_down_btn['bg'] = "white"
        
class Func2:
    def __init__(self, btn_frame, main_app):
        self.btn_frame = btn_frame
        self.main_app = main_app
        self.create_tab2_widgets()  # Create buttons and labels for Tab 1

    def create_tab2_widgets(self):
        self.ori_btn = ttk.Button(self.btn_frame, text="CLEAR", command=self.recover)
        self.ori_btn.grid(row=0, column=0, padx=10)

        self.show_image_difference_btn = tkmacosx.Button(self.btn_frame, text="Differ", command=lambda: change_color(self.show_image_difference_btn, image_difference), bg="white")
        self.show_image_difference_btn.grid(row=0, column=1, padx=10)

        self.enlarge_effect_fixed_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.enlarge_effect_fixed_btn, enlarge_effect), bg="white")
        self.enlarge_effect_fixed_btn.grid(row=0, column=2, padx=10)
        
        self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Reduce", command=lambda: change_color(self.reduce_effect_optimized_btn, reduce_effect), bg="white")
        self.reduce_effect_optimized_btn.grid(row=0, column=3, padx=10)

        self.apply_mosaic_effect_btn = tkmacosx.Button(self.btn_frame, text="Mosaic", command=lambda: change_color(self.apply_mosaic_effect_btn, mosaic_effect), bg="white")
        self.apply_mosaic_effect_btn.grid(row=0, column=4, padx=10)        

        # # self.enlarge_line_effect_btn = self.create_button("Enlarge Line Effect", lambda: enlarge_line_effect(self.img))
        # self.reduce_effect_optimized_btn = tkmacosx.Button(self.btn_frame, text="Enlarge", command=lambda: change_color(self.reduce_effect_optimized_btn, enlarge_effect), bg="white")
        # self.reduce_effect_optimized_btn.grid(row=0, column=2, padx=10)     
          
        self.grayscale_conversion_btn = tkmacosx.Button(self.btn_frame, text="Grayscale", command=lambda: change_color(self.grayscale_conversion_btn, grayscale_conversion), bg="white")
        self.grayscale_conversion_btn.grid(row=0, column=5, padx=10)

        self.sepia_tone_btn = tkmacosx.Button(self.btn_frame, text="Sepia", command=lambda: change_color(self.sepia_tone_btn, sepia_tone), bg="white")
        self.sepia_tone_btn.grid(row=0, column=6, padx=10)

        self.invert_colors_btn = tkmacosx.Button(self.btn_frame, text="Invert color", command=lambda: change_color(self.invert_colors_btn, invert_colors), bg="white")
        self.invert_colors_btn.grid(row=1, column=0, padx=10) 


        self.posterize_effect_btn = tkmacosx.Button(self.btn_frame, text="Poster", command=lambda: change_color(self.posterize_effect_btn, posterize_effect), bg="white")
        self.posterize_effect_btn.grid(row=1, column=1, padx=10)

        self.solarize_effect_btn = tkmacosx.Button(self.btn_frame, text="Solar", command=lambda: change_color(self.solarize_effect_btn, solarize_effect), bg="white")
        self.solarize_effect_btn.grid(row=1, column=2, padx=10)

        self.bitwise_not_effect_btn = tkmacosx.Button(self.btn_frame, text="Bitwise", command=lambda: change_color(self.bitwise_not_effect_btn, bitwise_not_effect), bg="white")
        self.bitwise_not_effect_btn.grid(row=1, column=3, padx=10)      

        self.emboss_effect_btn = tkmacosx.Button(self.btn_frame, text="Emboss", command=lambda: change_color(self.emboss_effect_btn, emboss_effect), bg="white")
        self.emboss_effect_btn.grid(row=1, column=4, padx=10)

        self.blur_effect_btn = tkmacosx.Button(self.btn_frame, text="Blur", command=lambda: change_color(self.blur_effect_btn, blur_effect), bg="white")
        self.blur_effect_btn.grid(row=1, column=5, padx=10)

        self.sharpen_effect_btn = tkmacosx.Button(self.btn_frame, text="Sharpen", command=lambda: change_color(self.sharpen_effect_btn, sharpen_effect), bg="white")
        self.sharpen_effect_btn.grid(row=1, column=6, padx=10)

        self.apply_oil_painting_btn = tkmacosx.Button(self.btn_frame, text="Oil paint", command=lambda: change_color(self.apply_oil_painting_btn, oil_painting), bg="white")
        self.apply_oil_painting_btn.grid(row=2, column=0, padx=10)   

        self.apply_sketch_effect_btn = tkmacosx.Button(self.btn_frame, text="Sketch", command=lambda: change_color(self.apply_sketch_effect_btn, sketch_effect), bg="white")
        self.apply_sketch_effect_btn.grid(row=2, column=1, padx=10)        

        self.apply_watercolor_effect_btn = tkmacosx.Button(self.btn_frame, text="Water color", command=lambda: change_color(self.apply_watercolor_effect_btn, watercolor_effect), bg="white")
        self.apply_watercolor_effect_btn.grid(row=2, column=2, padx=10)        

        self.apply_dreamy_effect_btn = tkmacosx.Button(self.btn_frame, text="Dreamy", command=lambda: change_color(self.apply_dreamy_effect_btn, dreamy_effect), bg="white")
        self.apply_dreamy_effect_btn.grid(row=2, column=3, padx=10)

        self.apply_pencil_sketch_btn = tkmacosx.Button(self.btn_frame, text="Pencil", command=lambda: change_color(self.apply_pencil_sketch_btn, pencil_sketch), bg="white")
        self.apply_pencil_sketch_btn.grid(row=2, column=4, padx=10)

        self.apply_cartoonize_btn = tkmacosx.Button(self.btn_frame, text="Cartoon", command=lambda: change_color(self.apply_cartoonize_btn, cartoonize), bg="white")
        self.apply_cartoonize_btn.grid(row=2, column=5, padx=10)

        self.apply_pixelate_btn = tkmacosx.Button(self.btn_frame, text="Pixel", command=lambda: change_color(self.apply_pixelate_btn, pixelate), bg="white")
        self.apply_pixelate_btn.grid(row=2, column=6, padx=10)

        self.apply_gaussian_blur_btn = tkmacosx.Button(self.btn_frame, text="G blur", command=lambda: change_color(self.apply_gaussian_blur_btn, gaussian_blur), bg="white")
        self.apply_gaussian_blur_btn.grid(row=3, column=0, padx=10)

        self.apply_halftone_btn = tkmacosx.Button(self.btn_frame, text="Halftone", command=lambda: change_color(self.apply_halftone_btn, halftone), bg="white")
        self.apply_halftone_btn.grid(row=3, column=1, padx=10)      

        self.apply_color_splash_btn = tkmacosx.Button(self.btn_frame, text="Splash", command=lambda: change_color(self.apply_color_splash_btn, color_splash), bg="white")
        self.apply_color_splash_btn.grid(row=3, column=2, padx=10)

        self.apply_vignette_btn = tkmacosx.Button(self.btn_frame, text="Vignette", command=lambda: change_color(self.apply_vignette_btn, vignette), bg="white")
        self.apply_vignette_btn.grid(row=3, column=3, padx=10)     

        self.apply_gradient_map_btn = tkmacosx.Button(self.btn_frame, text="Gradient", command=lambda: change_color(self.apply_gradient_map_btn, gradient_map), bg="white")
        self.apply_gradient_map_btn.grid(row=3, column=4, padx=10)

        self.apply_lens_flare_btn = tkmacosx.Button(self.btn_frame, text="Flare", command=lambda: change_color(self.apply_lens_flare_btn, lens_flare), bg="white")
        self.apply_lens_flare_btn.grid(row=3, column=5, padx=10)   

        self.apply_double_exposure_btn = tkmacosx.Button(self.btn_frame, text="Double exp", command=lambda: change_color(self.apply_double_exposure_btn, double_exposure), bg="white")
        self.apply_double_exposure_btn.grid(row=3, column=6, padx=10)

        self.apply_kaleidoscope_btn = tkmacosx.Button(self.btn_frame, text="Kalei", command=lambda: change_color(self.apply_kaleidoscope_btn, kaleidoscope), bg="white")
        self.apply_kaleidoscope_btn.grid(row=4, column=0, padx=10)   

        self.apply_glitch_art_btn = tkmacosx.Button(self.btn_frame, text="Glitch", command=lambda: change_color(self.apply_glitch_art_btn, glitch_art), bg="white")
        self.apply_glitch_art_btn.grid(row=4, column=1, padx=10)   


    def recover(self):
        image_difference.set(0)
        self.show_image_difference_btn['bg'] = "white"
        enlarge_effect.set(0)
        self.enlarge_effect_fixed_btn['bg'] = "white"
        reduce_effect.set(0)
        self.reduce_effect_optimized_btn['bg'] = "white"
        mosaic_effect.set(0)
        self.apply_mosaic_effect_btn['bg'] = "white"
        grayscale_conversion.set(0)
        self.grayscale_conversion_btn['bg'] = "white"
        sepia_tone.set(0)
        self.sepia_tone_btn['bg'] = "white"
        invert_colors.set(0)
        self.invert_colors_btn['bg'] = "white"
        posterize_effect.set(0)
        self.posterize_effect_btn['bg'] = "white"
        solarize_effect.set(0)
        self.solarize_effect_btn['bg'] = "white"
        bitwise_not_effect.set(0)
        self.bitwise_not_effect_btn['bg'] = "white"
        emboss_effect.set(0)
        self.emboss_effect_btn['bg'] = "white"
        blur_effect.set(0)
        self.blur_effect_btn['bg'] = "white"
        sharpen_effect.set(0)
        self.sharpen_effect_btn['bg'] = "white"
        oil_painting.set(0)
        self.apply_oil_painting_btn['bg'] = "white"
        sketch_effect.set(0)
        self.apply_sketch_effect_btn['bg'] = "white"
        watercolor_effect.set(0)
        self.apply_watercolor_effect_btn['bg'] = "white"
        dreamy_effect.set(0)
        self.apply_dreamy_effect_btn['bg'] = "white"
        pencil_sketch.set(0)
        self.apply_pencil_sketch_btn['bg'] = "white"
        cartoonize.set(0)
        self.apply_cartoonize_btn['bg'] = "white"
        pixelate.set(0)
        self.apply_pixelate_btn['bg'] = "white"
        gaussian_blur.set(0)
        self.apply_gaussian_blur_btn['bg'] = "white"

        halftone.set(0)
        self.apply_halftone_btn['bg'] = "white"

        color_splash.set(0)
        self.apply_color_splash_btn['bg'] = "white"

        vignette.set(0)
        self.apply_vignette_btn['bg'] = "white"
        
        gradient_map.set(0)
        self.apply_gradient_map_btn['bg'] = "white"

        lens_flare.set(0)
        self.apply_lens_flare_btn['bg'] = "white"

        double_exposure.set(0)
        self.apply_double_exposure_btn['bg'] = "white"

        kaleidoscope.set(0)
        self.apply_kaleidoscope_btn['bg'] = "white"

        glitch_art.set(0)
        self.apply_glitch_art_btn['bg'] = "white"

# Function to initialize global variables
def initialize_global_vars():
    global red_var, blue_var, green_var, zoom_in, zoom_out, flip_rl, flip_td, image_difference, enlarge_effect,reduce_effect
    global mosaic_effect, grayscale_conversion, invert_colors, sepia_tone, posterize_effect, solarize_effect, bitwise_not_effect
    global emboss_effect, blur_effect, sharpen_effect, oil_painting, sketch_effect, watercolor_effect, dreamy_effect
    global pencil_sketch, pixelate, cartoonize, gaussian_blur, halftone, color_splash, vignette, gradient_map, lens_flare
    global double_exposure, kaleidoscope, glitch_art
    red_var = tk.IntVar(value=0)
    blue_var = tk.IntVar(value=0)
    green_var = tk.IntVar(value=0)
    zoom_in = tk.IntVar(value=0)
    zoom_out = tk.IntVar(value=0)
    flip_rl = tk.IntVar(value=0)
    flip_td = tk.IntVar(value=0)
    image_difference = tk.IntVar(value=0)
    enlarge_effect = tk.IntVar(value=0)
    reduce_effect = tk.IntVar(value=0)
    mosaic_effect = tk.IntVar(value=0)
    # enlarge_line_effect = tk.IntVar(value=0) # still got problems.
    grayscale_conversion = tk.IntVar(value=0)
    sepia_tone = tk.IntVar(value=0)
    invert_colors = tk.IntVar(value=0)
    posterize_effect = tk.IntVar(value=0)
    solarize_effect = tk.IntVar(value=0)
    bitwise_not_effect = tk.IntVar(value=0)
    emboss_effect = tk.IntVar(value=0)
    blur_effect = tk.IntVar(value=0)
    sharpen_effect = tk.IntVar(value=0)
    oil_painting = tk.IntVar(value=0) # running too slow, need optimization
    sketch_effect = tk.IntVar(value=0)
    watercolor_effect = tk.IntVar(value=0)
    dreamy_effect = tk.IntVar(value=0)
    pencil_sketch = tk.IntVar(value=0)
    pixelate = tk.IntVar(value=0)
    cartoonize = tk.IntVar(value=0)
    gaussian_blur = tk.IntVar(value=0)
    halftone = tk.IntVar(value=0)
    color_splash = tk.IntVar(value=0)
    vignette = tk.IntVar(value=0)
    gradient_map = tk.IntVar(value=0)
    lens_flare = tk.IntVar(value=0)
    double_exposure = tk.IntVar(value=0)
    kaleidoscope = tk.IntVar(value=0)
    glitch_art = tk.IntVar(value=0)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
