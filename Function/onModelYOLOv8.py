import cv2
from ultralytics import YOLO

model = YOLO('../yolov8n.pt')
path = '../Images/third.jpg'
image = cv2.imread(path)

if image is None:
    print("Не удалось загрузить изображение.")
    exit()

results = model(image, classes=[9])
ramk = results[0].plot(conf=False, line_width=2, labels=False)

path2 = '../Images/detected.jpg'
cv2.imwrite(path2, ramk)

cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result', 800, 600)
cv2.moveWindow('Result', 300, 50)
cv2.imshow('Result', ramk)
cv2.waitKey(0)
cv2.destroyAllWindows()