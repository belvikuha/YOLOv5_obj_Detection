import torch
import cv2
import numpy as np
import pika
import io

from websocket import create_connection

# Загрузка модели
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

# Функция для распознавания и отображения объектов
def detect_objects(image_bytes):
    # Преобразование буфера в изображение
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_np, flags=cv2.IMREAD_COLOR)

    # Проверка, что изображение успешно загружено
    if img is None:
        print("Не удалось загрузить изображение")
        return

    # Обнаружение объектов
    results = model(img)

    # Визуализация результатов
    rendered_img = np.squeeze(results.render())

    # Кодирование изображения в формат JPEG для отправки по веб-сокету
    _, img_encoded = cv2.imencode('.jpg', rendered_img)
    return img_encoded.tobytes()

# Функция обработки сообщений из очереди
def callback(ch, method, properties, body):
    print("Получено изображение из очереди")
    processed_image = detect_objects(body)

    # Подключение к WebSocket и отправка обработанного изображения
    try:
        ws = create_connection("ws://localhost:8099")
        ws.send_binary(processed_image)
        ws.close()
    except Exception as e:
        print("Ошибка WebSocket:", e)

# Подключение к RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Создание очереди
channel.queue_declare(queue='imagesQueue')

# Установка обработчика сообщений
channel.basic_consume(queue='imagesQueue', on_message_callback=callback, auto_ack=True)

print('Ожидание изображений...')
channel.start_consuming()

