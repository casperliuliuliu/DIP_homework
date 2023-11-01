import tkinter as tk

def toggle_button():
    if toggle_var.get():
        toggle_label.config(text="Toggle: ON")
    else:
        toggle_label.config(text="Toggle: OFF")

# Create the main window
window = tk.Tk()
window.title("Toggle Button Example")

# Create a variable to store the state of the toggle button
toggle_var = tk.BooleanVar()
toggle_var.set(False)
# Create a label to display the toggle state
toggle_label = tk.Label(window, text="Toggle: OFF")
toggle_label.pack(pady=10)

# Create the toggle button
toggle_button = tk.Checkbutton(window, text="Toggle", variable=toggle_var, command=toggle_button)
toggle_button.pack()

# Run the Tkinter main loop
window.mainloop()
