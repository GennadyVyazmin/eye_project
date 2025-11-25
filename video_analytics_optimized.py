# video_analytics_optimized.py
import cv2
import numpy as np
import sqlite3
import datetime
import time
from deepface import DeepFace
from collections import defaultdict
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisitorCounter:
    def __init__(self, processing_interval=3.0, similarity_threshold=0.6):
        """
        Инициализация счетчика посетителей

        Args:
            processing_interval: интервал обработки лиц в секундах
            similarity_threshold: порог схожести лиц (0-1)
        """
        self.conn = sqlite3.connect('visitors.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
        self.similarity_threshold = similarity_threshold

        # Трекинг состояния
        self.last_processing_time = 0
        self.visitor_counter = 0
        self.current_frame_visitors = set()

        # Загрузка детектора лиц
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        logger.info("Инициализация VisitorCounter завершена")

    def _init_database(self):
        """Инициализация таблиц базы данных"""
        cursor = self.conn.cursor()

        # Таблица посетителей
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_embedding BLOB,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                visit_count INTEGER DEFAULT 1
            )
        ''')

        # Таблица почасовой статистики
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hour TIMESTAMP,
                unique_visitors INTEGER,
                total_detections INTEGER
            )
        ''')

        self.conn.commit()
        logger.info("База данных инициализирована")

    def get_face_embedding(self, face_image):
        """
        Получение эмбеддинга лица с оптимизацией

        Args:
            face_image: изображение лица

        Returns:
            embedding или None в случае ошибки
        """
        try:
            # Ресайз для ускорения обработки
            face_resized = cv2.resize(face_image, (160, 160))

            # Получение эмбеддинга
            result = DeepFace.represent(
                face_resized,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='opencv'
            )

            return result[0]['embedding']

        except Exception as e:
            logger.warning(f"Ошибка получения эмбеддинга: {e}")
            return None

    def compare_faces(self, embedding1, embedding2):
        """
        Сравнение двух эмбеддингов

        Args:
            embedding1: первый эмбеддинг
            embedding2: второй эмбеддинг

        Returns:
            bool: True если лица схожи
        """
        if embedding1 is None or embedding2 is None:
            return False

        try:
            distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
            similarity = 1 - distance
            return similarity > self.similarity_threshold
        except Exception as e:
            logger.warning(f"Ошибка сравнения лиц: {e}")
            return False

    def find_similar_visitor(self, embedding):
        """
        Поиск похожего посетителя в базе данных

        Args:
            embedding: эмбеддинг для поиска

        Returns:
            tuple: (visitor_id, similarity) или (None, 0) если не найден
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, face_embedding FROM visitors")
        visitors = cursor.fetchall()

        max_similarity = 0
        best_match_id = None

        for visitor_id, db_embedding_blob in visitors:
            if db_embedding_blob:
                try:
                    db_embedding = np.frombuffer(db_embedding_blob, dtype=np.float64)
                    distance = np.linalg.norm(np.array(embedding) - db_embedding)
                    similarity = 1 - distance

                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match_id = visitor_id
                except Exception as e:
                    logger.warning(f"Ошибка обработки эмбеддинга из БД: {e}")
                    continue

        return best_match_id, max_similarity

    def save_visitor(self, embedding):
        """
        Сохранение или обновление информации о посетителе

        Args:
            embedding: эмбеддинг лица

        Returns:
            int: ID посетителя
        """
        cursor = self.conn.cursor()
        now = datetime.datetime.now()

        # Поиск похожего лица в базе
        visitor_id, similarity = self.find_similar_visitor(embedding)

        if similarity > self.similarity_threshold:
            # Обновление существующего посетителя
            cursor.execute(
                "UPDATE visitors SET last_seen = ?, visit_count = visit_count + 1 WHERE id = ?",
                (now, visitor_id)
            )
            logger.info(f"Обновлен посетитель {visitor_id}, схожесть: {similarity:.3f}")

        else:
            # Добавление нового посетителя
            embedding_blob = np.array(embedding, dtype=np.float64).tobytes()
            cursor.execute(
                "INSERT INTO visitors (face_embedding, first_seen, last_seen, visit_count) VALUES (?, ?, ?, 1)",
                (embedding_blob, now, now)
            )
            visitor_id = cursor.lastrowid
            logger.info(f"Добавлен новый посетитель {visitor_id}")

        self.conn.commit()
        return visitor_id

    def process_detected_faces(self, frame, faces):
        """
        Обработка обнаруженных лиц в кадре

        Args:
            frame: исходный кадр
            faces: список обнаруженных лиц

        Returns:
            frame: обработанный кадр
        """
        current_embeddings = []
        processed_frame = frame.copy()

        for i, (x, y, w, h) in enumerate(faces):
            # Выделение области лица
            face_img = frame[y:y + h, x:x + w]

            # Пропускаем слишком маленькие лица
            if w < 30 or h < 30:
                continue

            # Получение эмбеддинга
            embedding = self.get_face_embedding(face_img)

            if embedding is not None:
                # Проверка на дубликаты в текущем кадре
                is_duplicate = False
                for existing_embedding in current_embeddings:
                    if self.compare_faces(embedding, existing_embedding):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    current_embeddings.append(embedding)

                    # Сохранение посетителя
                    visitor_id = self.save_visitor(embedding)

                    # Отрисовка bounding box и информации
                    color = (0, 255, 0)  # Зеленый
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(processed_frame, f'ID: {visitor_id}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Отображение схожести для отладки
                    cv2.putText(processed_frame, f'Face {i + 1}', (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return processed_frame, len(current_embeddings)

    def detect_faces(self, frame):
        """
        Обнаружение лиц в кадре

        Args:
            frame: входной кадр

        Returns:
            list: список координат лиц
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Детекция лиц с оптимизированными параметрами
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return faces

    def process_frame(self, frame):
        """
        Основная обработка кадра

        Args:
            frame: входной кадр

        Returns:
            tuple: (обработанный кадр, количество обработанных лиц)
        """
        current_time = time.time()

        # Обрабатываем кадр только если прошел достаточный интервал
        if current_time - self.last_processing_time < self.processing_interval:
            return frame, 0

        # Детекция лиц
        faces = self.detect_faces(frame)

        if len(faces) > 0:
            # Обработка обнаруженных лиц
            processed_frame, unique_faces_count = self.process_detected_faces(frame, faces)

            # Обновление времени последней обработки
            self.last_processing_time = current_time

            logger.info(f"Обработано лиц: {unique_faces_count} из {len(faces)} обнаруженных")

            return processed_frame, unique_faces_count
        else:
            return frame, 0

    def add_statistics_overlay(self, frame, faces_count, processed_faces):
        """
        Добавление статистической информации на кадр

        Args:
            frame: кадр для overlay
            faces_count: количество обнаруженных лиц
            processed_faces: количество обработанных лиц
        """
        # Фон для текста
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Статистика
        stats = [
            f"Detected: {faces_count}",
            f"Processed: {processed_faces}",
            f"Interval: {self.processing_interval}s",
            f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}"
        ]

        for i, text in enumerate(stats):
            cv2.putText(frame, text, (10, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def start_analysis(self, video_source=0, show_display=True):
        """
        Запуск анализа видео

        Args:
            video_source: источник видео (0 - камера, путь к файлу, URL)
            show_display: показывать ли окно с видео
        """
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            logger.error(f"Не удалось открыть видеоисточник: {video_source}")
            return

        logger.info(f"Анализ видео запущен (источник: {video_source})")

        frame_count = 0
        total_processed_faces = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Не удалось получить кадр")
                    break

                frame_count += 1

                # Обработка кадра
                processed_frame, processed_faces = self.process_frame(frame)
                total_processed_faces += processed_faces

                # Добавление статистики
                faces_in_frame = len(self.detect_faces(frame))
                self.add_statistics_overlay(processed_frame, faces_in_frame, processed_faces)

                # Отображение результата
                if show_display:
                    cv2.imshow('Visitor Analytics', processed_frame)

                    # Выход по нажатию 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Остановка по запросу пользователя")
                        break
                else:
                    # Вывод прогресса в консоль
                    if frame_count % 30 == 0:  # Каждые 30 кадров
                        logger.info(f"Кадр {frame_count}, всего обработано лиц: {total_processed_faces}")

        except KeyboardInterrupt:
            logger.info("Остановка по Ctrl+C")
        finally:
            cap.release()
            if show_display:
                cv2.destroyAllWindows()
            self.conn.close()
            logger.info(f"Анализ завершен. Всего кадров: {frame_count}, обработано лиц: {total_processed_faces}")

    def get_statistics(self):
        """Получение текущей статистики"""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM visitors")
        total_visitors = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(visit_count) FROM visitors")
        total_detections = cursor.fetchone()[0] or 0

        return {
            'total_visitors': total_visitors,
            'total_detections': total_detections
        }


def main():
    """Основная функция запуска"""
    import argparse

    parser = argparse.ArgumentParser(description='Система аналитики посетителей')
    parser.add_argument('--source', type=str, default='0',
                        help='Источник видео (0 - камера, путь к файлу)')
    parser.add_argument('--interval', type=float, default=3.0,
                        help='Интервал обработки лиц в секундах')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Порог схожести лиц (0-1)')
    parser.add_argument('--no-display', action='store_true',
                        help='Не показывать окно с видео')

    args = parser.parse_args()

    # Преобразование source в int если это число
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source

    # Создание и запуск счетчика
    counter = VisitorCounter(
        processing_interval=args.interval,
        similarity_threshold=args.threshold
    )

    try:
        counter.start_analysis(
            video_source=video_source,
            show_display=not args.no_display
        )
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        stats = counter.get_statistics()
        logger.info(f"Финальная статистика: {stats}")


if __name__ == "__main__":
    main()