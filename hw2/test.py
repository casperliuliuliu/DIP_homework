
import tkinter as tk

def toggle_state():
    global current_state
    if current_state.get() == 'A':
        current_state.set('B')
    else:
        current_state.set('A')

root = tk.Tk()
root.title("Switch Example")

current_state = tk.StringVar()
current_state.set('A')

label = tk.Label(root, text="Current State:")
label.pack(pady=10)

state_label = tk.Label(root, textvariable=current_state)
state_label.pack()

toggle_button = tk.Button(root, text="Toggle", command=toggle_state)
toggle_button.pack(pady=10)

root.mainloop()
