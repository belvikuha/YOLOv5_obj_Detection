import torch
import cv2
import numpy as np

# Загрузка модели
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

# Функция для распознавания и отображения объектов
def detect_objects(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path)

    # Проверка, что изображение успешно загружено
    if img is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return

    # Обнаружение объектов
    results = model(img)

    for i, (x, y, w, h, conf, cls) in enumerate(results.xywh[0]):
        label = results.names[int(cls)]
        print(f"Объект {i}: {label} (Уверенность: {conf:.2f}), Координаты: x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}")

    # Визуализация результатов
    cv2.imshow('YOLOv5 Detection', np.squeeze(results.render()))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Пример использования
image_path = 'image.jpg'  # Укажите путь к вашему изображению
detect_objects(image_path)
