
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime

class VideoApp:
    option = 'ori'
    recording = False
    filename = None
    recording_path = "/Users/liushiwen/Desktop/大四上/數位影像處理/DIP_homework/DIP_final/"
    def __init__(self, root, video_source=0, width=640, height=480):
        self.root = root
        self.root.title("Video Processing App")

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

        self.ori_btn = ttk.Button(self.btn_frame, text="Recover", command=self.recover)
        self.ori_btn.grid(row=0, column=0, padx=10)

        self.red_btn = ttk.Button(self.btn_frame, text="Red", command=self.apply_red_channel)
        self.red_btn.grid(row=0, column=1, padx=10)

        self.blue_btn = ttk.Button(self.btn_frame, text="Blue", command=self.apply_blue_channel)
        self.blue_btn.grid(row=0, column=2, padx=10)

        self.green_btn = ttk.Button(self.btn_frame, text="Green", command=self.apply_green_channel)
        self.green_btn.grid(row=0, column=3, padx=10)

        self.exit_btn = ttk.Button(self.btn_frame, text="Exit", command=self.exit_app)
        self.exit_btn.grid(row=0, column=4, padx=10)

        self.record_btn = ttk.Button(self.btn_frame, text="Record", command=self.start_recording)
        self.record_btn.grid(row=1, column=0, padx=10)

        self.stop_btn = ttk.Button(self.btn_frame, text="Stop", command=self.stop_recording)
        self.stop_btn.grid(row=1, column=1, padx=10)

        

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

        # Call the update method after 10 milliseconds
        self.root.after(10, self.video_update)
        
    def cool_filter(self, img):
        temp_img = img.copy()
        if self.option == "red":
            temp_img[:, :, 2] = 0  # Set blue channel to 0
            temp_img[:, :, 1] = 0  # Set green channel to 0
        elif self.option == "blue":
            temp_img[:, :, 0] = 0  # Set red channel to 0
            temp_img[:, :, 1] = 0  # Set green channel to 0
        elif self.option == "green":
            temp_img[:, :, 0] = 0  # Set red channel to 0
            temp_img[:, :, 2] = 0  # Set blue channel to 0
            
        img_right = Image.fromarray(temp_img)
        return img_right

    def recover(self):
        self.option = "ori"
    def apply_red_channel(self):
        self.option = "red"

    def apply_blue_channel(self):
        self.option = "blue"

    def apply_green_channel(self):
        self.option = "green"

    def exit_app(self):
        self.cap.release()
        self.root.destroy()

    def start_recording(self):
        self.recording = True
        self.filename = f"{self.recording_path}recording_{datetime.now()}.mp4"
        self.writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))

    def stop_recording(self):
        self.recording = False
        self.stop_btn["state"] = tk.DISABLED
        self.record_btn["state"] = tk.NORMAL
        self.writer.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
