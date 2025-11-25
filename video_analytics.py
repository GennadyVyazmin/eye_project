# video_analytics.py
import cv2
import numpy as np
import sqlite3
import datetime
import time
from deepface import DeepFace
import threading
from collections import defaultdict


class VisitorCounter:
    def __init__(self):
        self.conn = sqlite3.connect('visitors.db', check_same_thread=False)
        self.current_visitors = {}
        self.hourly_stats = defaultdict(int)
        self.last_hour_check = datetime.datetime.now()

        # Загрузка каскада для детекции лиц
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def get_face_embedding(self, face_image):
        """Получение эмбеддинга лица"""
        try:
            result = DeepFace.represent(face_image, model_name='Facenet', enforce_detection=False)
            return result[0]['embedding']
        except:
            return None

    def compare_faces(self, embedding1, embedding2, threshold=0.6):
        """Сравнение двух эмбеддингов"""
        if embedding1 is None or embedding2 is None:
            return False

        distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
        return distance < threshold

    def save_visitor(self, embedding):
        """Сохранение или обновление информации о посетителе"""
        cursor = self.conn.cursor()
        now = datetime.datetime.now()

        # Поиск похожего лица в базе
        cursor.execute("SELECT id, face_embedding, visit_count FROM visitors")
        visitors = cursor.fetchall()

        max_similarity = 0
        best_match_id = None

        for visitor_id, db_embedding_blob, visit_count in visitors:
            if db_embedding_blob:
                db_embedding = np.frombuffer(db_embedding_blob, dtype=np.float64)
                similarity = 1 - np.linalg.norm(np.array(embedding) - db_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_id = visitor_id

        if max_similarity > 0.6:  # Порог схожести
            # Обновление существующего посетителя
            cursor.execute(
                "UPDATE visitors SET last_seen = ?, visit_count = visit_count + 1 WHERE id = ?",
                (now, best_match_id)
            )
            visitor_id = best_match_id
        else:
            # Добавление нового посетителя
            embedding_blob = np.array(embedding).tobytes()
            cursor.execute(
                "INSERT INTO visitors (face_embedding, first_seen, last_seen) VALUES (?, ?, ?)",
                (embedding_blob, now, now)
            )
            visitor_id = cursor.lastrowid

        self.conn.commit()
        return visitor_id

    def update_hourly_stats(self):
        """Обновление почасовой статистики"""
        now = datetime.datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        if now - self.last_hour_check >= datetime.timedelta(hours=1):
            cursor = self.conn.cursor()

            # Подсчет уникальных посетителей за последний час
            one_hour_ago = now - datetime.timedelta(hours=1)
            cursor.execute(
                "SELECT COUNT(DISTINCT id) FROM visitors WHERE last_seen >= ?",
                (one_hour_ago,)
            )
            unique_visitors = cursor.fetchone()[0]

            # Сохранение статистики
            cursor.execute(
                "INSERT INTO hourly_stats (hour, unique_visitors, total_detections) VALUES (?, ?, ?)",
                (current_hour, unique_visitors, self.hourly_stats[current_hour])
            )

            self.conn.commit()
            self.hourly_stats.clear()
            self.last_hour_check = now

            print(f"Статистика за час {current_hour}: {unique_visitors} уникальных посетителей")

    def process_frame(self, frame):
        """Обработка кадра видео"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Выделение области лица
            face_img = frame[y:y + h, x:x + w]

            # Получение эмбеддинга
            embedding = self.get_face_embedding(face_img)

            if embedding is not None:
                visitor_id = self.save_visitor(embedding)

                # Рисование прямоугольника вокруг лица
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Visitor {visitor_id}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Обновление почасовой статистики
        self.update_hourly_stats()

        return frame

    def start_analysis(self, video_source=0):
        """Запуск анализа видео"""
        cap = cv2.VideoCapture(video_source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Обработка кадра
            processed_frame = self.process_frame(frame)

            # Отображение результата
            cv2.imshow('Visitor Counter', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.conn.close()


if __name__ == "__main__":
    counter = VisitorCounter()
    counter.start_analysis()