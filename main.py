import torch
import cv2
import numpy as np
import pika
import json
import base64
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

    keyWords = []
    for i, (x, y, w, h, conf, cls) in enumerate(results.xywh[0]):
        label = results.names[int(cls)]
        keyWords.append(label)
        print(f"Объект {i}: {label} (Уверенность: {conf:.2f}), Координаты: x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}")


    # Визуализация результатов
    # rendered_img = np.squeeze(results.render())

    # Кодирование изображения в формат JPEG для отправки по веб-сокету
    # _, img_encoded = cv2.imencode('.jpg', rendered_img)
    # return img_encoded.tobytes()
    return ', '.join(keyWords)

# Функция обработки сообщений из очереди
def callback(ch, method, properties, body):
    print("Получено изображение из очереди")
    # print(type(body))
    # print(body)
    # body_str = body.decode('utf-8')
    # deserialized_object = json.loads(body.decode('utf-8'))
    # print(deserialized_object)
    # print(type(deserialized_object))
    # image_bytes = bytes(deserialized_object['image'])
    # image_bytes = base64.b64decode(deserialized_object['image'])
    #
    # conectionId = deserialized_object['conectionId']
    # print(conectionId)
    processed_image = detect_objects(body)
    # Подключение к WebSocket и отправка обработанного изображения
    try:
        ws = create_connection("ws://localhost:8000")
        # ws.send_binary(processed_image)
        dataObject = {
            "message": processed_image,
            "method": "photokeywords receive",
            "id": 1,
        }

        ws.send(json.dumps(dataObject))
        ws.close()
    except Exception as e:
        print("Ошибка WebSocket:", e)

# Подключение к RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Создание очереди
channel.queue_declare(queue='imageKeyWord')

# Установка обработчика сообщений
channel.basic_consume(queue='imageKeyWord', on_message_callback=callback, auto_ack=True)

print('Ожидание изображений...')
channel.start_consuming()

