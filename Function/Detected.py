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