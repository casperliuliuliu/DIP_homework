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
            temp_img = apply_dreamy_effect(temp_img)
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
        
        # self.enlarge_effect_fixed_btn = self.create_button("Enlarge Effect Fixed", lambda: enlarge_effect_fixed(self.img))
        # self.reduce_effect_optimized_btn = self.create_button("Reduce Effect Optimized", lambda: reduce_effect_optimized(self.img))
        # self.apply_mosaic_effect_btn = self.create_button("Apply Mosaic Effect", lambda: apply_mosaic_effect(self.img))
        # self.enlarge_line_effect_btn = self.create_button("Enlarge Line Effect", lambda: enlarge_line_effect(self.img))
        # self.grayscale_conversion_btn = self.create_button("Grayscale Conversion", lambda: grayscale_conversion(self.img))
        # self.sepia_tone_btn = self.create_button("Sepia Tone", lambda: sepia_tone(self.img))
        # self.invert_colors_btn = self.create_button("Invert Colors", lambda: invert_colors(self.img))
        # self.posterize_effect_btn = self.create_button("Posterize Effect", lambda: posterize_effect(self.img))
        # self.solarize_effect_btn = self.create_button("Solarize Effect", lambda: solarize_effect(self.img, 128))
        # self.bitwise_not_effect_btn = self.create_button("Bitwise Not Effect", lambda: bitwise_not_effect(self.img))
        # self.emboss_effect_btn = self.create_button("Emboss Effect", lambda: emboss_effect(self.img))
        # self.blur_effect_btn = self.create_button("Blur Effect", lambda: blur_effect(self.img, 3))
        # self.sharpen_effect_btn = self.create_button("Sharpen Effect", lambda: sharpen_effect(self.img))
        # self.apply_oil_painting_btn = self.create_button("Apply Oil Painting", lambda: apply_oil_painting(self.img))
        # self.apply_sketch_effect_btn = self.create_button("Apply Sketch Effect", lambda: apply_sketch_effect(self.img))
        # self.apply_watercolor_effect_btn = self.create_button("Apply Watercolor Effect", lambda: apply_watercolor_effect(self.img))
        # self.apply_dreamy_effect_btn = self.create_button("Apply Dreamy Effect", lambda: apply_dreamy_effect(self.img))
        # self.apply_pencil_sketch_btn = self.create_button("Apply Pencil Sketch", lambda: apply_pencil_sketch(self.img))
        # self.apply_pixelate_btn = self.create_button("Apply Pixelate", lambda: apply_pixelate(self.img))
        # self.apply_cartoonize_btn = self.create_button("Apply Cartoonize", lambda: apply_cartoonize(self.img))
        # self.apply_gaussian_blur_btn = self.create_button("Apply Gaussian Blur", lambda: apply_gaussian_blur(self.img))
        # self.apply_halftone_btn = self.create_button("Apply Halftone", lambda: apply_halftone(self.img, 1))
        # self.apply_color_splash_btn = self.create_button("Apply Color Splash", lambda: apply_color_splash(self.img))
        # self.apply_vignette_btn = self.create_button("Apply Vignette", lambda: apply_vignette(self.img, 0.1))
        # self.apply_gradient_map_btn = self.create_button("Apply Gradient Map", lambda: apply_gradient_map(self.img))
        # self.apply_lens_flare_btn = self.create_button("Apply Lens Flare", lambda: apply_lens_flare(self.img))
        # self.apply_double_exposure_btn = self.create_button("Apply Double Exposure", lambda: apply_double_exposure(self.img))
        # self.apply_kaleidoscope_btn = self.create_button("Apply Kaleidoscope", lambda: apply_kaleidoscope(self.img, 10))
        # self.apply_glitch_art_btn = self.create_button("Apply Glitch Art", lambda: apply_glitch_art(self.img, 0.01))
        # self.enlarge_effect_fixed_btn = self.create_button("Enlarge Effect Fixed", lambda: enlarge_effect_fixed(self.img))
        # self.reduce_effect_optimized_btn = self.create_button("Reduce Effect Optimized", lambda: reduce_effect_optimized(self.img))
        # self.apply_mosaic_effect_btn = self.create_button("Apply Mosaic Effect", lambda: apply_mosaic_effect(self.img))
        # self.enlarge_line_effect_btn = self.create_button("Enlarge Line Effect", lambda: enlarge_line_effect(self.img))
        # self.grayscale_conversion_btn = self.create_button("Grayscale Conversion", lambda: grayscale_conversion(self.img))
        # self.sepia_tone_btn = self.create_button("Sepia Tone", lambda: sepia_tone(self.img))
        # self.invert_colors_btn = self.create_button("Invert Colors", lambda: invert_colors(self.img))
        # self.posterize_effect_btn = self.create_button("Posterize Effect", lambda: posterize_effect(self.img))
        # self.solarize_effect_btn = self.create_button("Solarize Effect", lambda: solarize_effect(self.img, 128))
        # self.bitwise_not_effect_btn = self.create_button("Bitwise Not Effect", lambda: bitwise_not_effect(self.img))
        # self.emboss_effect_btn = self.create_button("Emboss Effect", lambda: emboss_effect(self.img))
        # self.blur_effect_btn = self.create_button("Blur Effect", lambda: blur_effect(self.img, 3))
        # self.sharpen_effect_btn = self.create_button("Sharpen Effect", lambda: sharpen_effect(self.img))
        # self.apply_oil_painting_btn = self.create_button("Apply Oil Painting", lambda: apply_oil_painting(self.img))
        # self.apply_sketch_effect_btn = self.create_button("Apply Sketch Effect", lambda: apply_sketch_effect(self.img))
        # self.apply_watercolor_effect_btn = self.create_button("Apply Watercolor Effect", lambda: apply_watercolor_effect(self.img))
        # self.apply_dreamy_effect_btn = self.create_button("Apply Dreamy Effect", lambda: apply_dreamy_effect(self.img))
        # self.apply_pencil_sketch_btn = self.create_button("Apply Pencil Sketch", lambda: apply_pencil_sketch(self.img))
        # self.apply_pixelate_btn = self.create_button("Apply Pixelate", lambda: apply_pixelate(self.img))
        # self.apply_cartoonize_btn = self.create_button("Apply Cartoonize", lambda: apply_cartoonize(self.img))
        # self.apply_gaussian_blur_btn = self.create_button("Apply Gaussian Blur", lambda: apply_gaussian_blur(self.img))
        # self.apply_halftone_btn = self.create_button("Apply Halftone", lambda: apply_halftone(self.img, 1))
        # self.apply_color_splash_btn = self.create_button("Apply Color Splash", lambda: apply_color_splash(self.img))
        # self.apply_vignette_btn = self.create_button("Apply Vignette", lambda: apply_vignette(self.img, 0.1))
        # self.apply_gradient_map_btn = self.create_button("Apply Gradient Map", lambda: apply_gradient_map(self.img))
        # self.apply_lens_flare_btn = self.create_button("Apply Lens Flare", lambda: apply_lens_flare(self.img))
        # self.apply_double_exposure_btn = self.create_button("Apply Double Exposure", lambda: apply_double_exposure(self.img))
        # self.apply_kaleidoscope_btn = self.create_button("Apply Kaleidoscope", lambda: apply_kaleidoscope(self.img, 10))
        # self.apply_glitch_art_btn = self.create_button("Apply Glitch Art", lambda: apply_glitch_art(self.img, 0.01))

        # self.zoom_out_btn.grid(row=0, column=2, padx=10)
        # self.flip_right_to_left_btn.grid(row=0, column=3, padx=10)
        # self.flip_top_to_down_btn.grid(row=0, column=3, padx=10)
        # self.show_image_difference_btn.grid(row=0, column=4, padx=10)
        # self.enlarge_effect_fixed_btn.grid(row=0, column=5, padx=10)
        # self.reduce_effect_optimized_btn.grid(row=0, column=6, padx=10)
        # self.apply_mosaic_effect_btn.grid(row=0, column=7, padx=10)
        # self.enlarge_line_effect_btn.grid(row=0, column=8, padx=10)
        # self.grayscale_conversion_btn.grid(row=0, column=9, padx=10)
        # self.sepia_tone_btn.grid(row=0, column=10, padx=10)
        # self.invert_colors_btn.grid(row=0, column=11, padx=10)
        # self.posterize_effect_btn.grid(row=0, column=12, padx=10)
        # self.solarize_effect_btn.grid(row=0, column=13, padx=10)
        # self.bitwise_not_effect_btn.grid(row=0, column=14, padx=10)
        # self.emboss_effect_btn.grid(row=0, column=15, padx=10)
        # self.blur_effect_btn.grid(row=0, column=16, padx=10)
        # self.sharpen_effect_btn.grid(row=0, column=17, padx=10)
        # self.apply_oil_painting_btn.grid(row=0, column=18, padx=10)
        # self.apply_sketch_effect_btn.grid(row=0, column=19, padx=10)
        # self.apply_watercolor_effect_btn.grid(row=0, column=20, padx=10)
        # self.apply_dreamy_effect_btn.grid(row=0, column=21, padx=10)
        # self.apply_pencil_sketch_btn.grid(row=0, column=22, padx=10)
        # self.apply_pixelate_btn.grid(row=0, column=23, padx=10)
        # self.apply_cartoonize_btn.grid(row=0, column=24, padx=10)
        # self.apply_gaussian_blur_btn.grid(row=0, column=25, padx=10)
        # self.apply_halftone_btn.grid(row=0, column=26, padx=10)
        # self.apply_color_splash_btn.grid(row=0, column=27, padx=10)
        # self.apply_vignette_btn.grid(row=0, column=28, padx=10)
        # self.apply_gradient_map_btn.grid(row=0, column=29, padx=10)
        # self.apply_lens_flare_btn.grid(row=0, column=30, padx=10)
        # self.apply_double_exposure_btn.grid(row=0, column=31, padx=10)
        # self.apply_kaleidoscope_btn.grid(row=0, column=32, padx=10)
        # self.apply_glitch_art_btn.grid(row=0, column=33, padx=10)



    def recover(self):
        image_difference.set(0)
        self.show_image_difference_btn['bg'] = "white"
        # zoom_out.set(0)
        # self.zoom_out_btn['bg'] = "white"
        # flip_rl.set(0)
        # self.flip_right_to_left_btn['bg'] = "white"

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
