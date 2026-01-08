import cv2

def detect_traffic_light_state(roi):
    if roi.size == 0:
        return "unknown"

    h, w = roi.shape[:2]
    if h < 30 or w < 10:
        return "unknown"

    third = h // 3
    red_zone    = roi[0:third, :]
    yellow_zone = roi[third:2*third, :]
    green_zone  = roi[2*third:, :]

    def is_color_dominant(zone, lower, upper):
        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return cv2.countNonZero(mask) > (zone.shape[0] * zone.shape[1] * 0.1)

    red_lower1, red_upper1 = (0, 70, 50), (10, 255, 255)
    red_lower2, red_upper2 = (170, 70, 50), (180, 255, 255)
    yellow_lower, yellow_upper = (15, 70, 50), (35, 255, 255)
    green_lower, green_upper = (36, 70, 50), (85, 255, 255)

    red_mask1 = cv2.inRange(cv2.cvtColor(red_zone, cv2.COLOR_BGR2HSV), red_lower1, red_upper1)
    red_mask2 = cv2.inRange(cv2.cvtColor(red_zone, cv2.COLOR_BGR2HSV), red_lower2, red_upper2)
    red_active = (cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)) > (red_zone.shape[0] * red_zone.shape[1] * 0.1)

    yellow_active = is_color_dominant(yellow_zone, yellow_lower, yellow_upper)
    green_active = is_color_dominant(green_zone, green_lower, green_upper)

    if red_active:
        return "red"
    elif yellow_active:
        return "yellow"
    elif green_active:
        return "green"
    else:
        return "unknown"


def point_below_line(point, line_point1, line_point2):
    """
    Проверяет, находится ли точка ниже линии, заданной двумя точками.
    Для стоп-линии: точка считается "ниже", если она находится по направлению движения
    (обычно это означает, что Y координата точки больше Y координаты линии в этой точке X).
    
    Args:
        point: Точка (x, y)
        line_point1: Первая точка линии (x1, y1)
        line_point2: Вторая точка линии (x2, y2)
    
    Returns:
        bool: True если точка ниже линии (пересекла линию в направлении движения)
    """
    px, py = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    
    # Если линия вертикальная
    if x1 == x2:
        # Для вертикальной линии проверяем, находится ли точка справа от линии
        # (если линия идет сверху вниз) или слева (если линия идет снизу вверх)
        if y2 > y1:  # Линия идет сверху вниз
            return px > x1
        else:  # Линия идет снизу вверх
            return px < x1
    
    # Вычисляем уравнение прямой: y = k*x + b
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    
    # Y координата линии в точке px
    line_y = k * px + b
    
    # Точка ниже линии, если её Y больше Y линии
    # Это работает для горизонтальных и диагональных линий, идущих слева направо
    return py > line_y


def point_above_line(point, line_point1, line_point2):
    """
    Проверяет, находится ли точка выше линии, заданной двумя точками.
    Для стоп-линии: точка считается "выше", если она находится далеко за перекрестком
    (обычно это означает, что Y координата точки меньше Y координаты линии в этой точке X).
    
    Args:
        point: Точка (x, y)
        line_point1: Первая точка линии (x1, y1)
        line_point2: Вторая точка линии (x2, y2)
    
    Returns:
        bool: True если точка выше линии (далеко за перекрестком)
    """
    px, py = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    
    # Если линия вертикальная
    if x1 == x2:
        # Для вертикальной линии проверяем, находится ли точка слева от линии
        # (если линия идет сверху вниз) или справа (если линия идет снизу вверх)
        if y2 > y1:  # Линия идет сверху вниз
            return px < x1
        else:  # Линия идет снизу вверх
            return px > x1
    
    # Вычисляем уравнение прямой: y = k*x + b
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    
    # Y координата линии в точке px
    line_y = k * px + b
    
    # Точка выше линии, если её Y меньше Y линии
    return py < line_y


def line_intersects_box(line_point1, line_point2, box):
    """
    Проверяет, пересекает ли линия bounding box машины.
    
    Args:
        line_point1: Первая точка линии (x1, y1)
        line_point2: Вторая точка линии (x2, y2)
        box: Bounding box машины (x1, y1, x2, y2)
    
    Returns:
        bool: True если линия пересекает bounding box
    """
    x1, y1, x2, y2 = box
    lx1, ly1 = line_point1
    lx2, ly2 = line_point2
    
    # Углы bounding box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    # Проверяем положение углов относительно линии
    sides = []
    for corner in corners:
        below = point_below_line(corner, line_point1, line_point2)
        sides.append(below)
    
    # Если углы находятся по разные стороны линии, значит линия пересекает box
    if len(set(sides)) > 1:
        return True
    
    # Проверяем пересечение линии с границами box
    # Проверяем пересечение с левой границей
    if min(lx1, lx2) <= x1 <= max(lx1, lx2):
        if lx2 != lx1:
            k = (ly2 - ly1) / (lx2 - lx1)
            b = ly1 - k * lx1
            line_y_at_x1 = k * x1 + b
            if y1 <= line_y_at_x1 <= y2 or y2 <= line_y_at_x1 <= y1:
                return True
    
    # Проверяем пересечение с правой границей
    if min(lx1, lx2) <= x2 <= max(lx1, lx2):
        if lx2 != lx1:
            k = (ly2 - ly1) / (lx2 - lx1)
            b = ly1 - k * lx1
            line_y_at_x2 = k * x2 + b
            if y1 <= line_y_at_x2 <= y2 or y2 <= line_y_at_x2 <= y1:
                return True
    
    # Проверяем пересечение с верхней границей
    if min(ly1, ly2) <= y1 <= max(ly1, ly2):
        if lx2 != lx1:
            k = (ly2 - ly1) / (lx2 - lx1)
            b = ly1 - k * lx1
            line_x_at_y1 = (y1 - b) / k if k != 0 else None
            if line_x_at_y1 is not None and x1 <= line_x_at_y1 <= x2:
                return True
    
    # Проверяем пересечение с нижней границей
    if min(ly1, ly2) <= y2 <= max(ly1, ly2):
        if lx2 != lx1:
            k = (ly2 - ly1) / (lx2 - lx1)
            b = ly1 - k * lx1
            line_x_at_y2 = (y2 - b) / k if k != 0 else None
            if line_x_at_y2 is not None and x1 <= line_x_at_y2 <= x2:
                return True
    
    return False


def check_car_over_stop_line(car_boxes, stop_line_points, threshold=50):
    """
    Проверяет, пересекает ли машина стоп-линию или находится сразу за ней.
    
    Args:
        car_boxes: Список координат машин в формате [(x1, y1, x2, y2), ...]
        stop_line_points: Список из двух точек [(x1, y1), (x2, y2)] для стоп-линии
        threshold: Максимальное расстояние в пикселях для определения "сразу за линией" (по умолчанию 50)
    
    Returns:
        list: Список словарей с информацией о каждой машине:
            [{'box': (x1, y1, x2, y2), 'is_over': bool, 'distance': float}, ...]
    """
    if stop_line_points is None or len(stop_line_points) != 2:
        return []
    
    line_point1, line_point2 = stop_line_points
    results = []
    
    for box in car_boxes:
        if len(box) < 4:
            continue
        
        x1, y1, x2, y2 = box[:4]
        
        # Проверяем, пересекает ли стоп-линия bounding box машины
        intersects = line_intersects_box(line_point1, line_point2, box)
        
        # Проверяем нижнюю часть машины (центр нижней границы)
        car_bottom_center_x = (x1 + x2) // 2
        car_bottom_center_y = y2
        car_bottom_center = (car_bottom_center_x, car_bottom_center_y)
        
        # Проверяем, находится ли нижняя часть машины ниже линии
        bottom_below = point_below_line(car_bottom_center, line_point1, line_point2)
        
        # Вычисляем расстояние от нижней части машины до линии
        lx1, ly1 = line_point1
        lx2, ly2 = line_point2
        
        px, py = car_bottom_center
        
        # Вычисляем расстояние от точки до прямой
        if lx2 == lx1:
            # Вертикальная линия
            distance = abs(px - lx1)
        else:
            # Расстояние от точки до прямой: |Ax + By + C| / sqrt(A^2 + B^2)
            A = ly2 - ly1
            B = -(lx2 - lx1)
            C = (lx2 - lx1) * ly1 - (ly2 - ly1) * lx1
            distance = abs(A * px + B * py + C) / ((A**2 + B**2)**0.5)
        
        # Машина пересекает линию или находится сразу за ней, если:
        # 1. Линия пересекает bounding box машины, ИЛИ
        # 2. Нижняя часть машины находится ниже линии и расстояние не превышает threshold
        is_over = False
        if intersects:
            # Линия пересекает машину
            is_over = True
            distance = -distance if bottom_below else distance
        elif bottom_below and distance <= threshold:
            # Машина находится сразу за линией (в пределах threshold)
            is_over = True
            distance = -distance
        
        results.append({
            'box': (x1, y1, x2, y2),
            'is_over': is_over,
            'distance': distance
        })
    
    return results


def detect_cars_over_stopline(car_boxes, stop_line_points):
    """
    Определяет машины, заехавшие за стоп-линию, заданную двумя точками.
    
    Args:
        car_boxes: Список координат машин в формате [(x1, y1, x2, y2), ...]
        stop_line_points: Список из двух точек [(x1, y1), (x2, y2)] для стоп-линии (обязательный параметр)
    
    Returns:
        list: Список словарей с информацией о каждой машине
    """
    # Проверяем каждую машину
    results = check_car_over_stop_line(car_boxes, stop_line_points)
    
    return results