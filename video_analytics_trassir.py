# video_analytics_trassir.py
import cv2
import numpy as np
import sqlite3
import datetime
import time
from deepface import DeepFace
from collections import defaultdict
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrassirVisitorCounter:
    def __init__(self, processing_interval=0.3, similarity_threshold=0.65):
        """
        Инициализация счетчика посетителей для камеры Trassir
        """
        self.conn = sqlite3.connect('visitors_trassir.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
        self.similarity_threshold = similarity_threshold  # Высокий порог для качественной камеры

        # Трекинг состояния
        self.last_processing_time = 0
        self.known_visitors_cache = {}
        self.frame_count = 0

        # Загрузка детектора лиц - используем более точный детектор
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Предзагрузка известных посетителей
        self._load_known_visitors()

        logger.info(f"Инициализация для Trassir завершена. Порог схожести: {similarity_threshold}")

    def _init_database(self):
        """Инициализация базы данных"""
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_embedding BLOB,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                visit_count INTEGER DEFAULT 1,
                last_updated TIMESTAMP,
                quality_score REAL DEFAULT 1.0
            )
        ''')

        # Таблица для статистики по камере
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS camera_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                total_detections INTEGER,
                unique_visitors INTEGER,
                frame_quality REAL
            )
        ''')

        self.conn.commit()
        logger.info("База данных Trassir инициализирована")

    def _load_known_visitors(self):
        """Загрузка известных посетителей в кэш"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, face_embedding FROM visitors")
        visitors = cursor.fetchall()

        self.known_visitors_cache.clear()
        loaded_count = 0

        for visitor_id, embedding_blob in visitors:
            if embedding_blob:
                try:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float64)
                    self.known_visitors_cache[visitor_id] = embedding
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Ошибка загрузки посетителя {visitor_id}: {e}")

        logger.info(f"Загружено посетителей в кэш: {loaded_count}")

    def calculate_frame_quality(self, frame):
        """Оценка качества кадра для Trassir"""
        try:
            # Проверка резкости через лапласиан
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Проверка яркости
            brightness = np.mean(gray)

            # Проверка контраста
            contrast = np.std(gray)

            quality_score = min(1.0, laplacian_var / 1000.0)  # Нормализация

            return quality_score
        except:
            return 0.5

    def get_face_embedding(self, face_image):
        """Получение эмбеддинга лица с улучшенной обработкой для Trassir"""
        try:
            # Увеличиваем размер для использования деталей высокого разрешения
            face_resized = cv2.resize(face_image, (224, 224))

            # Улучшенная предобработка для профессиональной камеры
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Улучшение контраста и яркости
            lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            face_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

            result = DeepFace.represent(
                face_enhanced,
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
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            similarity = np.dot(emb1_norm, emb2_norm)
            return float(similarity)
        except Exception as e:
            logger.warning(f"Ошибка расчета схожести: {e}")
            return 0.0

    def find_best_match(self, embedding):
        """Поиск лучшего совпадения среди известных посетителей"""
        best_match_id = None
        best_similarity = 0.0

        for visitor_id, known_embedding in self.known_visitors_cache.items():
            similarity = self.calculate_similarity(embedding, known_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = visitor_id

        return best_match_id, best_similarity

    def save_visitor(self, embedding, quality_score=1.0):
        """Сохранение или обновление информации о посетителе"""
        cursor = self.conn.cursor()
        now = datetime.datetime.now()

        visitor_id, similarity = self.find_best_match(embedding)

        if similarity > self.similarity_threshold:
            # Обновление существующего посетителя
            cursor.execute(
                """UPDATE visitors SET last_seen = ?, visit_count = visit_count + 1, 
                   last_updated = ?, quality_score = ? WHERE id = ?""",
                (now, now, quality_score, visitor_id)
            )
            self.known_visitors_cache[visitor_id] = embedding
            logger.info(f"🔄 ОБНОВЛЕН посетитель {visitor_id}, схожесть: {similarity:.3f}")

        else:
            # Добавление нового посетителя
            embedding_blob = embedding.tobytes()
            cursor.execute(
                """INSERT INTO visitors (face_embedding, first_seen, last_seen, 
                   visit_count, last_updated, quality_score) VALUES (?, ?, ?, 1, ?, ?)""",
                (embedding_blob, now, now, now, quality_score)
            )
            visitor_id = cursor.lastrowid
            self.known_visitors_cache[visitor_id] = embedding
            logger.info(f"🆕 НОВЫЙ посетитель {visitor_id}, схожесть: {similarity:.3f}")

        self.conn.commit()
        return visitor_id

    def _process_multiple_faces(self, face_data, processed_frame, frame_quality):
        """Обработка нескольких лиц для Trassir"""
        processed_count = 0
        embeddings_cache = {}

        # Получаем эмбеддинги для всех лиц
        for i, (x, y, w, h, face_img) in enumerate(face_data):
            try:
                embedding = self.get_face_embedding(face_img)
                if embedding is not None:
                    embeddings_cache[i] = (x, y, w, h, embedding)
            except Exception as e:
                logger.warning(f"Ошибка получения эмбеддинга: {e}")

        # Обработка с проверкой дубликатов
        processed_embeddings = []

        for i, (x, y, w, h, embedding) in embeddings_cache.items():
            # Проверка на дубликаты в текущем кадре
            is_duplicate_in_frame = False
            for existing_embedding in processed_embeddings:
                if self.calculate_similarity(embedding, existing_embedding) > 0.8:
                    is_duplicate_in_frame = True
                    break

            if not is_duplicate_in_frame:
                visitor_id = self.save_visitor(embedding, frame_quality)
                processed_embeddings.append(embedding)
                processed_count += 1

                # Определяем цвет рамки
                best_match_id, similarity = self.find_best_match(embedding)
                is_new = similarity <= self.similarity_threshold

                color = (0, 0, 255) if is_new else (0, 255, 0)
                status = "NEW" if is_new else "KNOWN"

                # Отрисовка с улучшенной визуализацией
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(processed_frame, f'{status}: {visitor_id}', (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(processed_frame, f'Visits: {self.get_visit_count(visitor_id)}',
                            (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(processed_frame, f'Sim: {similarity:.2f}',
                            (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return processed_count

    def process_frame(self, frame):
        """Обработка кадра для Trassir с оптимизацией под высокое разрешение"""
        current_time = time.time()

        if current_time - self.last_processing_time < self.processing_interval:
            return frame, 0, 0, 0.0

        # Оценка качества кадра
        frame_quality = self.calculate_frame_quality(frame)

        # Ресайз для ускорения обработки (сохраняя детализацию)
        height, width = frame.shape[:2]
        if width > 1920:
            scale = 1920 / width
            new_width = 1920
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame

        # Детекция лиц с оптимизированными параметрами для Trassir
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Более точный поиск
            minNeighbors=8,  # Меньше ложных срабатываний
            minSize=(80, 80),  # Больший минимальный размер
            maxSize=(400, 400),  # Ограничение для крупных планов
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        detected_count = len(faces)
        processed_count = 0

        if detected_count > 0:
            processed_frame = frame_resized.copy()
            face_data = []

            # Масштабируем координаты обратно если делали ресайз
            scale_x = width / processed_frame.shape[1]
            scale_y = height / processed_frame.shape[0]

            for (x, y, w, h) in faces:
                # Корректировка координат для исходного размера
                x_orig = int(x * scale_x)
                y_orig = int(y * scale_y)
                w_orig = int(w * scale_x)
                h_orig = int(h * scale_y)

                if w_orig < 60 or h_orig < 60 or w_orig > 500 or h_orig > 500:
                    continue

                face_img = frame[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]
                face_data.append((x_orig, y_orig, w_orig, h_orig, face_img))

            # Обрабатываем все лица
            processed_count = self._process_multiple_faces(face_data, processed_frame, frame_quality)

            self.last_processing_time = current_time
            self.frame_count += 1

            logger.info(
                f"Кадр {self.frame_count}: Обнаружено: {detected_count}, Обработано: {processed_count}, Качество: {frame_quality:.2f}")
            return processed_frame, detected_count, processed_count, frame_quality

        return frame, 0, 0, frame_quality

    def get_visit_count(self, visitor_id):
        """Получение количества визитов посетителя"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT visit_count FROM visitors WHERE id = ?", (visitor_id,))
        result = cursor.fetchone()
        return result[0] if result else 1

    def setup_rtsp_camera(self, rtsp_url):
        """Настройка подключения к RTSP камере Trassir"""
        cap = cv2.VideoCapture(rtsp_url)

        # Настройки для RTPS потока
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

        # Даем камере время на инициализацию
        time.sleep(2)

        return cap

    def start_analysis(self, rtsp_url):
        """Запуск анализа с RTPS камеры Trassir"""
        logger.info(f"Подключение к камере Trassir: {rtsp_url}")

        cap = self.setup_rtsp_camera(rtsp_url)

        if not cap.isOpened():
            logger.error(f"Не удалось подключиться к камере: {rtsp_url}")
            return

        logger.info(f"🚀 Анализ Trassir запущен. RTSP: {rtsp_url}")
        logger.info(f"Всего посетителей в базе: {len(self.known_visitors_cache)}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Потеряно соединение с камерой, переподключение...")
                    cap.release()
                    time.sleep(5)
                    cap = self.setup_rtsp_camera(rtsp_url)
                    if not cap.isOpened():
                        logger.error("Не удалось переподключиться к камере")
                        break
                    continue

                processed_frame, detected, processed, quality = self.process_frame(frame)

                # Расширенная статистика
                stats_text = [
                    f"TRASSIR CAMERA - 2K",
                    f"Detected: {detected}",
                    f"Processed: {processed}",
                    f"Total in DB: {len(self.known_visitors_cache)}",
                    f"Quality: {quality:.2f}",
                    f"Threshold: {self.similarity_threshold}",
                    f"Frame: {self.frame_count}",
                    f"Press 'q' to quit"
                ]

                # Фон для текста
                overlay = processed_frame.copy()
                cv2.rectangle(overlay, (0, 0), (400, 200), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, processed_frame, 0.4, 0, processed_frame)

                for i, text in enumerate(stats_text):
                    cv2.putText(processed_frame, text, (10, 25 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow('Trassir Visitor Analytics - 2K QUALITY', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Остановка по запросу пользователя")
                    break

        except KeyboardInterrupt:
            logger.info("Остановка по Ctrl+C")
        except Exception as e:
            logger.error(f"Ошибка во время анализа: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.conn.close()
            logger.info(f"✅ Анализ Trassir завершен. Обработано кадров: {self.frame_count}")

    def cleanup_database(self):
        """Очистка базы данных для тестирования"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM visitors")
        cursor.execute("DELETE FROM camera_stats")
        self.conn.commit()
        self.known_visitors_cache.clear()
        logger.info("🗑️ База данных Trassir очищена")


def main():
    """Основная функция для Trassir камеры"""

    # RTSP URL вашей камеры Trassir
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    # Создаем счетчик оптимизированный для Trassir
    counter = TrassirVisitorCounter(
        processing_interval=0.3,  # Частая обработка для плавности
        similarity_threshold=0.65  # Высокий порог для качественной камеры
    )

    # Раскомментируйте для очистки базы:
    # counter.cleanup_database()

    try:
        counter.start_analysis(RTSP_URL)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        logger.info("Работа с камерой Trassir завершена")


if __name__ == "__main__":
    main()