# video_analytics_fixed.py
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
    def __init__(self, processing_interval=5.0, similarity_threshold=0.7):
        """
        Инициализация счетчика посетителей

        Args:
            processing_interval: интервал обработки лиц в секундах
            similarity_threshold: порог схожести лиц (0-1), ВЫШЕ = строже
        """
        self.conn = sqlite3.connect('visitors.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
        self.similarity_threshold = similarity_threshold

        # Трекинг состояния
        self.last_processing_time = 0
        self.known_visitors_cache = {}  # Кэш известных посетителей для ускорения

        # Загрузка детектора лиц
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Предзагрузка известных посетителей
        self._load_known_visitors()

        logger.info(f"Инициализация завершена. Порог схожести: {similarity_threshold}")

    def _init_database(self):
        """Инициализация таблиц базы данных"""
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_embedding BLOB,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                visit_count INTEGER DEFAULT 1,
                last_updated TIMESTAMP
            )
        ''')

        self.conn.commit()

    def _load_known_visitors(self):
        """Загрузка известных посетителей в кэш"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, face_embedding FROM visitors")
        visitors = cursor.fetchall()

        self.known_visitors_cache.clear()
        for visitor_id, embedding_blob in visitors:
            if embedding_blob:
                try:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float64)
                    self.known_visitors_cache[visitor_id] = embedding
                except Exception as e:
                    logger.warning(f"Ошибка загрузки посетителя {visitor_id}: {e}")

        logger.info(f"Загружено посетителей в кэш: {len(self.known_visitors_cache)}")

    def get_face_embedding(self, face_image):
        """Получение эмбеддинга лица"""
        try:
            # Ресайз для ускорения
            face_resized = cv2.resize(face_image, (160, 160))

            # Конвертация в RGB (DeepFace ожидает RGB)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            result = DeepFace.represent(
                face_rgb,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='opencv'
            )

            return np.array(result[0]['embedding'], dtype=np.float64)

        except Exception as e:
            logger.warning(f"Ошибка получения эмбеддинга: {e}")
            return None

    def calculate_similarity(self, embedding1, embedding2):
        """Расчет схожести между эмбеддингами"""
        if embedding1 is None or embedding2 is None:
            return 0.0

        try:
            # Нормализация эмбеддингов
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)

            # Косинусное сходство (лучше для FaceNet)
            similarity = np.dot(emb1_norm, emb2_norm)
            return float(similarity)
        except Exception as e:
            logger.warning(f"Ошибка расчета схожести: {e}")
            return 0.0

    def find_best_match(self, embedding):
        """
        Поиск лучшего совпадения среди известных посетителей

        Returns:
            tuple: (visitor_id, similarity) или (None, 0) если не найден
        """
        best_match_id = None
        best_similarity = 0.0

        for visitor_id, known_embedding in self.known_visitors_cache.items():
            similarity = self.calculate_similarity(embedding, known_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = visitor_id

        logger.debug(f"Лучшее совпадение: ID {best_match_id}, схожесть: {best_similarity:.3f}")
        return best_match_id, best_similarity

    def save_visitor(self, embedding):
        """
        Сохранение или обновление информации о посетителе
        """
        cursor = self.conn.cursor()
        now = datetime.datetime.now()

        # Поиск лучшего совпадения
        visitor_id, similarity = self.find_best_match(embedding)

        if similarity > self.similarity_threshold:
            # ОБНОВЛЕНИЕ существующего посетителя
            cursor.execute(
                "UPDATE visitors SET last_seen = ?, visit_count = visit_count + 1, last_updated = ? WHERE id = ?",
                (now, now, visitor_id)
            )

            # Обновление кэша
            self.known_visitors_cache[visitor_id] = embedding

            logger.info(f"🔄 ОБНОВЛЕН посетитель {visitor_id}, схожесть: {similarity:.3f}")

        else:
            # ДОБАВЛЕНИЕ нового посетителя (только если действительно новый)
            embedding_blob = embedding.tobytes()
            cursor.execute(
                "INSERT INTO visitors (face_embedding, first_seen, last_seen, visit_count, last_updated) VALUES (?, ?, ?, 1, ?)",
                (embedding_blob, now, now, now)
            )
            visitor_id = cursor.lastrowid

            # Добавление в кэш
            self.known_visitors_cache[visitor_id] = embedding

            logger.info(f"🆕 НОВЫЙ посетитель {visitor_id}, схожесть: {similarity:.3f}")

        self.conn.commit()
        return visitor_id

    def process_frame(self, frame):
        """
        Обработка кадра с улучшенной логикой
        """
        current_time = time.time()

        # Пропускаем кадр если не прошел интервал
        if current_time - self.last_processing_time < self.processing_interval:
            return frame, 0, 0

        # Детекция лиц
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,  # Более строгий параметр
            minSize=(50, 50),  # Больший минимальный размер
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        detected_count = len(faces)
        processed_count = 0

        if detected_count > 0:
            processed_frame = frame.copy()
            current_embeddings = []

            for (x, y, w, h) in faces:
                # Пропускаем слишком маленькие/большие лица
                if w < 50 or h < 50 or w > 300 or h > 300:
                    continue

                face_img = frame[y:y + h, x:x + w]
                embedding = self.get_face_embedding(face_img)

                if embedding is not None:
                    # Проверка на дубликаты в текущем кадре
                    is_duplicate = False
                    for existing_embedding in current_embeddings:
                        if self.calculate_similarity(embedding,
                                                     existing_embedding) > 0.8:  # Высокий порог для дубликатов в кадре
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        current_embeddings.append(embedding)
                        visitor_id = self.save_visitor(embedding)
                        processed_count += 1

                        # Отрисовка
                        color = (0, 255, 0) if processed_count > 0 else (0, 0, 255)
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(processed_frame, f'ID: {visitor_id}', (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self.last_processing_time = current_time
            logger.info(
                f"Обнаружено: {detected_count}, Обработано: {processed_count}, Всего в базе: {len(self.known_visitors_cache)}")
            return processed_frame, detected_count, processed_count

        return frame, 0, 0

    def start_analysis(self, video_source=0):
        """Запуск анализа"""
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            logger.error(f"Не удалось открыть видеоисточник: {video_source}")
            return

        logger.info(f"🚀 Анализ запущен. Всего посетителей в базе: {len(self.known_visitors_cache)}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, detected, processed = self.process_frame(frame)

                # Добавление статистики
                stats_text = [
                    f"Detected: {detected}",
                    f"Processed: {processed}",
                    f"Total in DB: {len(self.known_visitors_cache)}",
                    f"Interval: {self.processing_interval}s",
                    f"Threshold: {self.similarity_threshold}"
                ]

                for i, text in enumerate(stats_text):
                    cv2.putText(processed_frame, text, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Visitor Analytics - FIXED', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Остановка по Ctrl+C")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.conn.close()
            logger.info(f"✅ Анализ завершен. Всего уникальных посетителей: {len(self.known_visitors_cache)}")

    def cleanup_database(self):
        """Очистка базы данных от дубликатов (для тестирования)"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM visitors")
        self.conn.commit()
        self.known_visitors_cache.clear()
        logger.info("🗑️ База данных очищена")


def main():
    """Основная функция с настройками для исправления проблемы"""

    # Создаем счетчик с более строгими параметрами
    counter = VisitorCounter(
        processing_interval=5.0,  # Увеличили интервал
        similarity_threshold=0.7  # Повысили порог схожести
    )

    # Очистка базы для тестирования (раскомментируйте если нужно)
    # counter.cleanup_database()

    try:
        counter.start_analysis(0)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        logger.info("Работа завершена")


if __name__ == "__main__":
    main()