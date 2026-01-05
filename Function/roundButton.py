import tkinter as tk

def create_rounded_button(parent, text, command, BG_COLOR, BTN_BG, BTN_FG, BTN_HOVER, width=160, height=40, radius=12):
    # Создаём Canvas как контейнер
    canvas = tk.Canvas(
        parent, width=width, height=height,
        bg=BG_COLOR, highlightthickness=0, relief="flat"
    )

    canvas.create_oval(0, 0, radius * 2, radius * 2, fill=BTN_BG, outline=BTN_BG)
    canvas.create_oval(width - radius * 2, 0, width, radius * 2, fill=BTN_BG, outline=BTN_BG)
    canvas.create_oval(0, height - radius * 2, radius * 2, height, fill=BTN_BG, outline=BTN_BG)
    canvas.create_oval(width - radius * 2, height - radius * 2, width, height, fill=BTN_BG, outline=BTN_BG)
    canvas.create_rectangle(radius, 0, width - radius, height, fill=BTN_BG, outline=BTN_BG)
    canvas.create_rectangle(0, radius, width, height - radius, fill=BTN_BG, outline=BTN_BG)

    canvas.create_text(width // 2, height // 2, text=text, fill=BTN_FG, font=("Segoe UI", 10, "bold"))

    def on_enter(e):
        canvas.itemconfig("all", fill=BTN_HOVER)
        canvas.itemconfig("text", fill=BTN_FG)

    def on_leave(e):
        canvas.itemconfig("all", fill=BTN_BG)
        canvas.itemconfig("text", fill=BTN_FG)

    def on_click(e):
        command()

    canvas.delete("all")
    canvas.create_oval(0, 0, radius * 2, radius * 2, fill=BTN_BG, outline=BTN_BG, tags="bg")
    canvas.create_oval(width - radius * 2, 0, width, radius * 2, fill=BTN_BG, outline=BTN_BG, tags="bg")
    canvas.create_oval(0, height - radius * 2, radius * 2, height, fill=BTN_BG, outline=BTN_BG, tags="bg")
    canvas.create_oval(width - radius * 2, height - radius * 2, width, height, fill=BTN_BG, outline=BTN_BG, tags="bg")
    canvas.create_rectangle(radius, 0, width - radius, height, fill=BTN_BG, outline=BTN_BG, tags="bg")
    canvas.create_rectangle(0, radius, width, height - radius, fill=BTN_BG, outline=BTN_BG, tags="bg")
    canvas.create_text(width // 2, height // 2, text=text, fill=BTN_FG, font=("Segoe UI", 10, "bold"), tags="text")

    canvas.tag_bind("bg", "<Enter>", on_enter)
    canvas.tag_bind("text", "<Enter>", on_enter)
    canvas.tag_bind("bg", "<Leave>", on_leave)
    canvas.tag_bind("text", "<Leave>", on_leave)
    canvas.tag_bind("bg", "<Button-1>", on_click)
    canvas.tag_bind("text", "<Button-1>", on_click)

    return canvas