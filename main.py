import cv2
import numpy as np
import serial
import time
import sys
import os
from serial.tools import list_ports


# ---------- утилита для PyInstaller (ищем файлы внутри .exe) ----------
def resource_path(relative_path):
    """
    При запуске как .py берёт файлы из текущей папки,
    при запуске как .exe (PyInstaller) — из временной.
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ---------- поиск порта Arduino ----------
def find_arduino_port():
    ports = list_ports.comports()
    for p in ports:
        desc = p.description.lower()
        if "arduino" in desc or "ch340" in desc or "usb-serial" in desc:
            return p.device
    return None


# ---------- связь с Arduino ----------
def connect_arduino(baudrate=9600):
    port = find_arduino_port()
    if port is None:
        print("❌ Arduino не найден. Подключи плату и перезапусти программу.")
        return None
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        print(f"✅ Подключено к Arduino на порту {port}")
        return ser
    except Exception as e:
        print("❌ Не удалось открыть порт Arduino:", e)
        return None


def send(ser, cmd):
    if ser is None:
        return
    try:
        ser.write((cmd + "\n").encode("utf-8"))
    except Exception as e:
        print("Ошибка отправки в Arduino:", e)


# ---------- поиск камеры ----------
def open_camera():
    """
    Пытаемся найти рабочую камеру с индексами 0..4.
    На Windows используем DirectShow (CAP_DSHOW),
    чтобы лучше цеплять USB-вебки.
    """
    is_windows = os.name == "nt"

    for idx in range(5):
        if is_windows:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            print(f"✅ Камера найдена, индекс {idx}")
            return cap

        cap.release()

    print("❌ Не удалось открыть ни одну камеру (индексы 0..4). " 
          "Проверь, что другая программа не использует вебкамеру.")
    return None


# ---------- загрузка YOLO ----------
cfg_path = resource_path(os.path.join("models", "yolov3-tiny.cfg"))
weights_path = resource_path(os.path.join("models", "yolov3-tiny.weights"))
names_path = resource_path(os.path.join("models", "coco.names"))

net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

TARGET_CLASSES = ["car", "truck", "bus"]
ln = net.getUnconnectedOutLayersNames()

# ---------- старт ----------
ser = connect_arduino()
current_light = "R"
send(ser, "R")

# открываем камеру
cap = open_camera()
if cap is None:
    sys.exit(1)

print("▶ Запуск TrafficAI. Нажми 'q' в окне видео для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Нет кадра с камеры")
        break

    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    cars_count = 0
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.4 and classes[class_id] in TARGET_CLASSES:
                box = detection[0:4] * np.array([W, H, W, H])
                (cX, cY, w, h) = box.astype("int")
                x = int(cX - w/2)
                y = int(cY - h/2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cars_count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # --- логика светофора ---
    desired = "G" if cars_count > 0 else "R"

    if current_light == "G" and desired == "R":
        print("Машин нет → мигаем жёлтым и переходим на красный")
        send(ser, "Y")
        time.sleep(2.2)
        current_light = "R"
    elif desired != current_light:
        print(f"Меняем свет: {current_light} → {desired}")
        send(ser, desired)
        current_light = desired

    cv2.putText(frame, f"Cars: {cars_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("TrafficAI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if ser is not None:
    ser.close()
cv2.destroyAllWindows()
