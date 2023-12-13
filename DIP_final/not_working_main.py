import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime
import tkmacosx
from funcs import change_color

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")
        # Create a ttk.Notebook to hold the tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=0, column=0, columnspan=6)

        # Create a frame for the first tab
        self.tab1_frame = ttk.Frame(self.notebook)
        self.tab1_functions = Func1(self.tab1_frame, self)

        # Create a frame for the second tab
        self.tab2_frame = ttk.Frame(self.notebook)
        self.tab2_functions = Func2(self.tab2_frame, self)

        # Create a frame for the second tab
        # self.tab3_frame = ttk.Frame(self.notebook)
        # self.tab3_functions = Func3(self.tab3_frame, self)

        # Add the frames as tabs to the notebook
        self.notebook.add(self.tab1_frame, text="Homework 1")
        self.notebook.add(self.tab2_frame, text="Homework 2")
        # self.notebook.add(self.tab3_frame, text="Homework 3")
        # self.create_tab1_widgets()  # Create buttons and labels for Tab 1
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
    
        if self.red_var.get():
            temp_img[:, :, 2] = 0  # Set blue channel to 0
            temp_img[:, :, 1] = 0  # Set green channel to 0
        if self.blue_var.get():
            temp_img[:, :, 0] = 0  # Set red channel to 0
            temp_img[:, :, 1] = 0  # Set green channel to 0
        if self.green_var.get():
            temp_img[:, :, 0] = 0  # Set red channel to 0
            temp_img[:, :, 2] = 0  # Set blue channel to 0
                # Call the update method after 10 milliseconds
        self.root.after(10, self.video_update)

        img_right = Image.fromarray(temp_img)
        return img_right
    
class VideoApp:
    option = 'ori'
    recording = False
    filename = None
    # recording_path = "/Users/liushiwen/Desktop/大四上/數位影像處理/DIP_homework/DIP_final/"
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

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)

        self.tab1_frame = ttk.Frame(self.btn_frame)
        self.tab1_functions = Func1(self.tab1_frame, self)

        # Create a frame for the second tab
        self.tab2_frame = ttk.Frame(self.btn_frame)
        self.tab2_functions = Func2(self.tab2_frame, self)

        # # Create a frame for the second tab
        # self.tab3_frame = ttk.Frame(self.notebook)
        # self.tab3_functions = Func3(self.tab3_frame, self)

        # Add the frames as tabs to the notebook
        self.notebook.add(self.tab1_frame, text="Homework 1")
        self.notebook.add(self.tab2_frame, text="Homework 2")
        # self.notebook.add(self.tab3_frame, text="Homework 3")


class Func1:
    def __init__(self, btn_frame, main_app):
        self.btn_frame = btn_frame
        self.main_app = main_app
        self.create_tab1_widgets()  # Create buttons and labels for Tab 1

    def create_tab1_widgets(self):

        self.ori_btn = ttk.Button(self.btn_frame, text="Recover", command=self.recover)
        self.ori_btn.grid(row=0, column=0, padx=10)

        self.red_var = tk.IntVar(value=0)
        self.red_btn = tkmacosx.Button(self.btn_frame, text="Red", command=lambda: change_color(self.red_btn, self.red_var), bg="white")
        self.red_btn.grid(row=0, column=1, padx=10)

        self.blue_var = tk.IntVar(value=0)
        self.blue_btn = tkmacosx.Button(self.btn_frame, text="Blue", command=lambda: change_color(self.blue_btn, self.blue_var), bg="white")
        self.blue_btn.grid(row=0, column=2, padx=10)

class Func2:
    def __init__(self, btn_frame, main_app):
        self.btn_frame = btn_frame
        self.main_app = main_app
        self.create_tab2_widgets()  # Create buttons and labels for Tab 1

    def create_tab2_widgets(self):
        self.green_var = tk.IntVar(value=0)
        self.green_btn = tkmacosx.Button(self.btn_frame, text="Green", command=lambda: change_color(self.green_btn, self.green_var), bg="white")
        self.green_btn.grid(row=0, column=3, padx=10)

        self.exit_btn = ttk.Button(self.btn_frame, text="Exit", command=self.exit_app)
        self.exit_btn.grid(row=0, column=4, padx=10)

        self.record_btn = ttk.Button(self.btn_frame, text="Record", command=self.start_recording)
        self.record_btn.grid(row=1, column=0, padx=10)

        self.stop_btn = ttk.Button(self.btn_frame, text="Stop", command=self.stop_recording)
        self.stop_btn.grid(row=1, column=1, padx=10)




    def recover(self):
        self.red_var.set(0)
        self.red_btn['bg'] = "white"
        self.blue_var.set(0)
        self.blue_btn['bg'] = "white"
        self.green_var.set(0)
        self.green_btn['bg'] = "white"


    def exit_app(self):
        self.cap.release()
        self.root.destroy()


    def start_recording(self):
        # self.recording = True
        # self.filename = f"{self.recording_path}recording_{datetime.now()}.mp4"
        # self.writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))
        pass

    def stop_recording(self):
        # self.recording = False
        # self.stop_btn["state"] = tk.DISABLED
        # self.record_btn["state"] = tk.NORMAL
        # self.writer.release()
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
