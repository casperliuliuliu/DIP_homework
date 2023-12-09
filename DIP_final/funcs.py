def change_color(button, intvar):
    # Get current color
    current_color = button["bg"]
    # Toggle between white and yellow
    if current_color == "white":
        new_color = "orange"
        intvar.set(1)
    else:
        new_color = "white"
        intvar.set(0)
    # Change the button's background color
    button["bg"] = new_color