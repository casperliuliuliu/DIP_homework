import tkinter as tk
from tkinter import ttk
from hw1 import Func1
from hw2 import Func2
from hw3_source import Func3

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
        self.tab3_frame = ttk.Frame(self.notebook)
        self.tab3_functions = Func3(self.tab3_frame, self)

        # Add the frames as tabs to the notebook
        self.notebook.add(self.tab1_frame, text="Homework 1")
        self.notebook.add(self.tab2_frame, text="Homework 2")
        self.notebook.add(self.tab3_frame, text="Homework 3")
        # self.create_tab1_widgets()  # Create buttons and labels for Tab 1
            
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()
