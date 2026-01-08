import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import time
from roundButton import create_rounded_button
from Detected import detect_cars_over_stop_line, detect_traffic_light_state

model = None
image = None
detected_image = None
original_cv2 = None
stop_line_points = []
image_scale_info = {}

def load_model(model_path="yolo12n.pt"):
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
        "YOLO12n (быстро)": "yolo12n.pt",
        "YOLO12s (баланс)": "yolo12s.pt",
        "YOLO12m (точно)": "yolo12m.pt"
    }
    if selected in model_map:
        load_model(model_map[selected])


def open_image():
    global image, original_cv2, detected_image, stop_line_points

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
        stop_line_points = []

        show_image(pil_img, canvas_before)
        clear_canvas(canvas_after)
        status_label.config(
            text="Изображение загружено. Кликните дважды на изображении, чтобы установить стоп-линию (2 точки)",
            fg="#b993d6")

    except Exception as e:
        messagebox.showerror("Ошибка", "Не удалось открыть изображение")

def set_stop_line(event):
    global stop_line_points, image_scale_info, original_cv2, image

    if original_cv2 is None or image is None:
        return

    canvas_id = id(canvas_before)
    if canvas_id not in image_scale_info:
        return

    scale_info = image_scale_info[canvas_id]

    canvas_x = event.x
    canvas_y = event.y

    relative_x = canvas_x - scale_info['offset_x']
    relative_y = canvas_y - scale_info['offset_y']

    if relative_x < 0:
        relative_x = 0
    if relative_y < 0:
        relative_y = 0

    img_x = int(relative_x / scale_info['ratio'])
    img_y = int(relative_y / scale_info['ratio'])

    h, w = original_cv2.shape[:2]
    if img_x < 0:
        img_x = 0
    elif img_x >= w:
        img_x = w - 1
    if img_y < 0:
        img_y = 0
    elif img_y >= h:
        img_y = h - 1

    if len(stop_line_points) == 0:
        stop_line_points = [(img_x, img_y)]
        status_label.config(text=f"Первая точка установлена ({img_x}, {img_y}). Кликните ещё раз для второй точки",
                            fg="#b993d6")
    elif len(stop_line_points) == 1:
        stop_line_points.append((img_x, img_y))
        status_label.config(text=f"Стоп-линия установлена. Нажмите 'Проверить стоп-линию'", fg="lightgreen")
    else:
        stop_line_points = [(img_x, img_y)]
        status_label.config(text=f"Первая точка установлена ({img_x}, {img_y}). Кликните ещё раз для второй точки",
                            fg="#b993d6")


    img_rgb = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2RGB)
    annotated = img_rgb.copy()

    for point in stop_line_points:
        cv2.circle(annotated, point, 5, (255, 0, 0), -1)

    if len(stop_line_points) == 2:
        cv2.line(annotated, stop_line_points[0], stop_line_points[1], (255, 0, 0), 3)
        cv2.putText(annotated, "STOP LINE", (stop_line_points[0][0], stop_line_points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    pil_annotated = Image.fromarray(annotated)
    show_image(pil_annotated, canvas_before)


def detect_cars_over_stop_line():
    global image, detected_image, original_cv2, stop_line_points

    if original_cv2 is None:
        messagebox.showwarning("Внимание", "Сначала загрузите изображение!")
        return

    if model is None:
        messagebox.showwarning("Внимание", "Модель ещё не загружена!")
        return

    if len(stop_line_points) != 2:
        messagebox.showwarning("Внимание", "Сначала установите стоп-линию!")
        return

    try:
        img_preprocessed = preprocess_image(original_cv2, kernel_size=3)
        start_time = time.time()

        car_results_model = model(img_preprocessed, classes=[2], verbose=False)

        traffic_light_results = model(img_preprocessed, classes=[9], verbose=False)

        elapsed_time = time.time() - start_time
        annotated_cv2 = original_cv2.copy()

        car_boxes_list = car_results_model[0].boxes
        if len(car_boxes_list) == 0:
            status_label.config(text="Машина не найдена", fg="#ffb347")
            show_image(image, canvas_after)
            detected_image = image.copy()
            return

        car_boxes = []
        for box in car_boxes_list:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            car_boxes.append((x1, y1, x2, y2))

        traffic_light_state = "unknown"
        traffic_light_boxes = traffic_light_results[0].boxes

        if len(traffic_light_boxes) > 0:
            tl_box = traffic_light_boxes[0]
            tl_x1, tl_y1, tl_x2, tl_y2 = map(int, tl_box.xyxy[0])
            tl_roi = original_cv2[tl_y1:tl_y2, tl_x1:tl_x2]
            traffic_light_state = detect_traffic_light_state(tl_roi)

            color_map = {
                "red": (0, 0, 255),
                "yellow": (0, 255, 255),
                "green": (0, 255, 0),
                "unknown": (128, 128, 128)
            }
            tl_color = color_map.get(traffic_light_state, (128, 128, 128))
            cv2.rectangle(annotated_cv2, (tl_x1, tl_y1), (tl_x2, tl_y2), tl_color, 3)
            cv2.putText(annotated_cv2, f"TRAFFIC LIGHT: {traffic_light_state.upper()}",
                        (tl_x1, tl_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tl_color, 2)

        car_results = detect_cars_over_stop_line(car_boxes, stop_line_points)

        cv2.line(annotated_cv2, stop_line_points[0], stop_line_points[1], (0, 0, 255), 3)
        cv2.circle(annotated_cv2, stop_line_points[0], 5, (0, 0, 255), -1)
        cv2.circle(annotated_cv2, stop_line_points[1], 5, (0, 0, 255), -1)
        cv2.putText(annotated_cv2, "STOP LINE", (stop_line_points[0][0], stop_line_points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cars_over_count = 0
        for i, result in enumerate(car_results):
            x1, y1, x2, y2 = result['box']
            is_over = result['is_over']

            is_violation = False
            if is_over:
                if traffic_light_state == "red":
                    is_violation = True
                elif traffic_light_state == "green":
                    is_violation = False
                else:
                    is_violation = False

            if is_violation:
                cv2.rectangle(annotated_cv2, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated_cv2, "VIOLATION", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cars_over_count += 1
            elif is_over:
                cv2.rectangle(annotated_cv2, (x1, y1), (x2, y2), (0, 250, 0), 3)
            else:
                cv2.rectangle(annotated_cv2, (x1, y1), (x2, y2), (0, 250, 0), 3)

        annotated_rgb = cv2.cvtColor(annotated_cv2, cv2.COLOR_BGR2RGB)
        pil_result = Image.fromarray(annotated_rgb)

        detected_image = pil_result.copy()
        show_image(pil_result, canvas_after)

        traffic_light_info = f", светофор: {traffic_light_state.upper()}" if traffic_light_state != "unknown" else ""
        status_label.config(
            text=f"Обнаружено нарушений: {cars_over_count}",
            fg="lightgreen" if cars_over_count == 0 else "#ff6b6b"
        )
        print(f"Время {elapsed_time:.3f} секунд")

    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось выполнить детекцию: {str(e)}")


def show_image(pil_image, canvas):
    global image_scale_info

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

    canvas_id = id(canvas)
    image_scale_info[canvas_id] = {
        'ratio': ratio,
        'img_w': img_w,
        'img_h': img_h,
        'canvas_w': canvas_w,
        'canvas_h': canvas_h,
        'offset_x': (canvas_w - new_w) // 2,
        'offset_y': (canvas_h - new_h) // 2
    }


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
        detected_image.save(filepath)
        messagebox.showinfo("Успех", "Изображение сохранено")
        status_label.config(text="Результат сохранён", fg="lightgreen")

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
    values=["YOLO12n (быстро)", "YOLO12s (баланс)", "YOLO12m (точно)"],
    state="readonly",
    width=18,
    font=("Segoe UI", 10)
)
model_selector.set("YOLO12n (быстро)")
model_selector.pack(side="left", padx=(0, 15))
model_selector.bind("<<ComboboxSelected>>", switch_model)

style = ttk.Style()
style.theme_use('clam')
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

btn_open_canvas = create_rounded_button(top_frame, "Открыть изображение", open_image, BG_COLOR, BTN_BG, BTN_FG,
                                        BTN_HOVER)
btn_open_canvas.pack(side="left", padx=(0, 15))

btn_check_stop_line = create_rounded_button(top_frame, "Обнаружить нарушение", detect_cars_over_stop_line, BG_COLOR,
                                            BTN_BG, BTN_FG, BTN_HOVER)
btn_check_stop_line.pack(side="left", padx=(0, 15))

btn_save_canvas = create_rounded_button(top_frame, "Сохранить результат", save_result, BG_COLOR, BTN_BG, BTN_FG,
                                        BTN_HOVER)
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

canvas_before = tk.Canvas(left_frame, bg="#2d2d3a", highlightthickness=0, cursor="crosshair")
canvas_before.pack(fill="both", expand=True, padx=5, pady=5)
canvas_before.bind("<Button-1>", set_stop_line)

right_frame = tk.LabelFrame(
    main_frame, text="Результат",
    bg=FRAME_BG, fg="#e0e0e0",
    font=("Segoe UI", 10, "bold"),
    relief="flat", bd=0
)
right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

canvas_after = tk.Canvas(right_frame, bg="#2d2d3a", highlightthickness=0)
canvas_after.pack(fill="both", expand=True, padx=5, pady=5)

root.after(100, lambda: load_model("yolo12s.pt"))
root.mainloop()