import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.open_button = tk.Button(root, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=5)

        self.save_button = tk.Button(root, text="Save Image", command=self.save_image)
        self.save_button.pack(pady=5)

        self.image = None

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.tif *.tiff")])
        if file_path:
            self.image = Image.open(file_path)
            self.display_image()

    def save_image(self):
        if self.image:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg *.jpeg"), ("TIF files", "*.tif *.tiff")])
            if file_path:
                self.image.save(file_path)
                messagebox.showinfo("Save Image", "Image saved successfully!")

    def display_image(self):
        if self.image:
            # self.image = self.image.convert("L")  # Convert to grayscale
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()
