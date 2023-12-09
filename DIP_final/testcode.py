import tkinter as tk
from tkmacosx import Button
def change_color(button, intvar):
    # Get current color
    current_color = button["bg"]
    # Toggle between white and yellow
    if current_color == "white":
        new_color = "orange"
        intvar = 1
    else:
        new_color = "white"
        intvar = 0
    # Change the button's background color
    button["bg"] = new_color
    print(intvar)

root = tk.Tk()
intvar = 0
button = Button(root, text="Toggle Color", command=lambda: change_color(button, intvar), bg="white")
button.pack()
root.mainloop()
