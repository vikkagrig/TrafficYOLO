import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import time
from roundButton import create_rounded_button

model = None
image = None
detected_image = None
original_cv2 = None

def load_model(model_path="yolov10n.pt"):
    global model, current_model_name
    try:
        status_label.config(text=f"Загружаем {model_path}...", fg="#b993d6")
        root.update()
        model = YOLO(model_path)
        current_model_name = model_path
        status_label.config(text=f"Модель {model_path} загружена", fg="lightgreen")
    except Exception as e:
        status_label.config(text=f"Ошибка загрузки {model_path}", fg="#ff6b6b")
        print(f"Ошибка: {e}")

def switch_model(event=None):
    selected = model_selector.get()
    model_map = {
        "YOLOv10n (быстро)": "yolov10n.pt",
        "YOLOv10s (баланс)": "yolov10s.pt",
        "YOLOv10m (точно)": "yolov10m.pt"
    }
    if selected in model_map:
        load_model(model_map[selected])

def open_image():
    global image, original_cv2, detected_image

    filepath = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not filepath:
        return

    try:
        img_cv2 = cv2.imread(filepath)
        if img_cv2 is None:
            raise ValueError("Невозможно прочитать изображение")

        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        image = pil_img.copy()
        original_cv2 = img_cv2.copy()
        detected_image = None

        show_image(pil_img, canvas_before)
        clear_canvas(canvas_after)
        status_label.config(text="Изображение загружено. Нажмите 'Найти светофор'", fg="#b993d6")

    except Exception as e:
        messagebox.showerror("Ошибка", "Не удалось открыть изображение")


def detect_traffic_light():
    global image, detected_image, original_cv2

    if original_cv2 is None:
        messagebox.showwarning("Внимание", "Сначала загрузите изображение!")
        return

    if model is None:
        messagebox.showwarning("Внимание", "Модель ещё не загружена!")
        return

    try:
        img_preprocessed = preprocess_image(original_cv2, kernel_size=3)
        start_time = time.time()
        results = model(img_preprocessed, classes=[9], verbose=False)
        elapsed_time = time.time() - start_time
        annotated_cv2 = original_cv2.copy()

        boxes = results[0].boxes
        if len(boxes) == 0:
            status_label.config(text="Светофор не найден", fg="#ffb347")
            show_image(image, canvas_after)
            detected_image = image.copy()
            return

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_cv2, (x1, y1), (x2, y2), (0, 250, 0), 3)

        annotated_rgb = cv2.cvtColor(annotated_cv2, cv2.COLOR_BGR2RGB)
        pil_result = Image.fromarray(annotated_rgb)

        detected_image = pil_result.copy()
        show_image(pil_result, canvas_after)

        found_count = len(boxes)
        status_label.config(text=f"Найдено светофоров: {found_count}", fg="lightgreen")
        print(f"Время {elapsed_time:.3f} секунд")

    except Exception as e:
        messagebox.showerror("Ошибка", "Не удалось выполнить детекцию")


def show_image(pil_image, canvas):
    canvas_w = canvas.winfo_width()
    canvas_h = canvas.winfo_height()
    if canvas_w < 10 or canvas_h < 10:
        canvas_w, canvas_h = 400, 300

    img_w, img_h = pil_image.size
    ratio = min(canvas_w / img_w, canvas_h / img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)

    resized = pil_image.resize((new_w, new_h), Image.LANCZOS)
    photo = ImageTk.PhotoImage(resized)

    canvas.delete("all")
    canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor="center")
    canvas.image = photo


def clear_canvas(canvas):
    canvas.delete("all")


def save_result():
    global detected_image

    if detected_image is None:
        messagebox.showwarning("Внимание", "Нет изображения для сохранения")
        return

    filepath = filedialog.asksaveasfilename(
        title="Сохранить результат",
        defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
    )
    if not filepath:
        return

    try:
        results = model(original_cv2, classes=[9], verbose=False)
        annotated = original_cv2.copy()
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 250, 0), 3)

        is_saved = cv2.imwrite(filepath, annotated)
        if is_saved:
            messagebox.showinfo("Успех", "Изображение сохранено")
            status_label.config(text="Результат сохранён", fg="lightgreen")
        else:
            raise Exception()

    except:
        messagebox.showerror("Ошибка", "Не удалось сохранить файл")

def preprocess_image(img_cv2, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным числом")

    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_denoised = cv2.medianBlur(img_rgb, kernel_size)

    return img_denoised

root = tk.Tk()
root.title("Обнаружение светофоров")
root.geometry("1000x600")
root.minsize(800, 500)

BG_COLOR = "#1e1e2e"
FRAME_BG = "#252536"
STATUS_BG = "#2d2d44"
BTN_BG = "#b993d6"
BTN_HOVER = "#a87fd1"
BTN_FG = "#1e1e2e"

root.configure(bg=BG_COLOR)

top_frame = tk.Frame(root, bg=BG_COLOR)
top_frame.pack(fill="x", padx=20, pady=20)

model_label = tk.Label(
    top_frame, text="Модель:",
    bg=BG_COLOR, fg="#cccccc", font=("Segoe UI", 10)
)
model_label.pack(side="left", padx=(0, 5))

model_selector = ttk.Combobox(
    top_frame,
    values=["YOLOv8n (быстро)", "YOLOv8s (баланс)", "YOLOv8m (точно)"],
    state="readonly",
    width=18,
    font=("Segoe UI", 10)
)
model_selector.set("YOLOv8n (быстро)")
model_selector.pack(side="left", padx=(0, 15))
model_selector.bind("<<ComboboxSelected>>", switch_model)

# Стилизация Combobox под тёмную тему
style = ttk.Style()
style.theme_use('clam')  # используем современную тему
style.configure(
    "TCombobox",
    fieldbackground=STATUS_BG,
    background=STATUS_BG,
    foreground="#cccccc",
    selectbackground=BTN_BG,
    selectforeground=BTN_FG,
    arrowcolor="#cccccc"
)

root.option_add('*TCombobox*Listbox.background', FRAME_BG)
root.option_add('*TCombobox*Listbox.foreground', '#cccccc')
root.option_add('*TCombobox*Listbox.selectBackground', BTN_BG)
root.option_add('*TCombobox*Listbox.selectForeground', BTN_FG)

btn_open_canvas = create_rounded_button(top_frame, "Открыть изображение", open_image, BG_COLOR, BTN_BG, BTN_FG, BTN_HOVER)
btn_open_canvas.pack(side="left", padx=(0, 15))

btn_detect_canvas = create_rounded_button(top_frame, "Найти светофор", detect_traffic_light, BG_COLOR, BTN_BG, BTN_FG, BTN_HOVER)
btn_detect_canvas.pack(side="left", padx=(0, 15))

btn_save_canvas = create_rounded_button(top_frame, "Сохранить результат", save_result, BG_COLOR, BTN_BG, BTN_FG, BTN_HOVER)
btn_save_canvas.pack(side="left")

status_label = tk.Label(
    top_frame, text="Загружаем модель",
    bg=STATUS_BG, fg="#cccccc", font=("Segoe UI", 10),
    padx=16, pady=8, relief="flat"
)
status_label.pack(side="right", padx=(20, 0))

main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

left_frame = tk.LabelFrame(
    main_frame, text="Исходное изображение",
    bg=FRAME_BG, fg="#e0e0e0",
    font=("Segoe UI", 10, "bold"),
    relief="flat", bd=0
)
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

canvas_before = tk.Canvas(left_frame, bg="#2d2d3a", highlightthickness=0)
canvas_before.pack(fill="both", expand=True, padx=5, pady=5)

right_frame = tk.LabelFrame(
    main_frame, text="Результат",
    bg=FRAME_BG, fg="#e0e0e0",
    font=("Segoe UI", 10, "bold"),
    relief="flat", bd=0
)
right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

canvas_after = tk.Canvas(right_frame, bg="#2d2d3a", highlightthickness=0)
canvas_after.pack(fill="both", expand=True, padx=5, pady=5)

root.after(100, lambda: load_model("yolov10m.pt"))
root.mainloop()