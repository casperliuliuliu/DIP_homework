import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime
import tkmacosx
from funcs import change_color

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

        self.ori_btn = ttk.Button(self.btn_frame, text="Recover", command=self.recover)
        self.ori_btn.grid(row=0, column=0, padx=10)

        self.red_btn = tkmacosx.Button(self.btn_frame, text="Red", command=lambda: change_color(self.red_btn, red_var), bg="white")
        self.red_btn.grid(row=0, column=1, padx=10)

        self.blue_btn = tkmacosx.Button(self.btn_frame, text="Blue", command=lambda: change_color(self.blue_btn, blue_var), bg="white")
        self.blue_btn.grid(row=0, column=2, padx=10)

        self.green_btn = tkmacosx.Button(self.btn_frame, text="Green", command=lambda: change_color(self.green_btn, green_var), bg="white")
        self.green_btn.grid(row=0, column=3, padx=10)

    def recover(self):
        red_var.set(0)
        self.red_btn['bg'] = "white"
        blue_var.set(0)
        self.blue_btn['bg'] = "white"
        green_var.set(0)
        self.green_btn['bg'] = "white"

class Func2:
    def __init__(self, btn_frame, main_app):
        self.btn_frame = btn_frame
        self.main_app = main_app
        self.create_tab2_widgets()  # Create buttons and labels for Tab 1

    def create_tab2_widgets(self):
        self.ori_btn = ttk.Button(self.btn_frame, text="Recover", command=self.recover)
        self.ori_btn.grid(row=0, column=0, padx=10)



    def recover(self):
        pass

# Function to initialize global variables
def initialize_global_vars():
    global red_var, blue_var, green_var
    red_var = tk.IntVar(value=0)
    blue_var = tk.IntVar(value=0)
    green_var = tk.IntVar(value=0)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
