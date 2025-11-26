# video_analytics_trassir_final.py
import cv2
import numpy as np
import sqlite3
import datetime
import time
from deepface import DeepFace
import logging
import threading
from queue import Queue
from collections import deque, defaultdict
import os
import shutil
import math
from threading import Lock

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FastTrassirCounter:
    def __init__(self, processing_interval=1.5, similarity_threshold=0.65, tracking_threshold=0.50):
        """
        БЫСТРАЯ версия с большим окном и минимальными задержками
        """
        self.conn = sqlite3.connect('visitors_trassir_fast.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
        self.similarity_threshold = similarity_threshold
        self.tracking_threshold = tracking_threshold

        # Мьютексы для потокобезопасности
        self.stats_lock = Lock()
        self.tracks_lock = Lock()

        # Цвета для индикации статусов
        self.COLORS = {
            'detected': (0, 255, 0),  # Зеленый - лицо обнаружено
            'tracking': (255, 255, 0),  # Желтый - создан трек
            'known': (0, 255, 255),  # Голубой - известный пользователь
            'new': (0, 0, 255),  # Красный - новый пользователь в БД
            'analyzing': (255, 165, 0),  # Оранжевый - анализ в процессе
            'rejected': (128, 128, 128)  # Серый - отсеян ложный объект
        }

        # Папки для хранения фото
        self.photos_dir = "visitor_photos_fast"
        self.current_session_dir = "current_session"
        self._create_directories()

        # Трекинг состояния
        self.last_processing_time = 0
        self.known_visitors_cache = {}
        self.frame_count = 0

        # Система трекинга лиц
        self.face_tracks = {}
        self.next_track_id = 1
        self.track_max_age = 5.0  # Уменьшили время жизни трека

        # Статистика
        self.recognition_stats = {
            'total_detections': 0,
            'valid_detections': 0,
            'rejected_detections': 0,
            'new_visitors': 0,
            'known_visitors': 0,
            'frames_processed': 0,
            'faces_processed': 0,
            'quality_rejections': defaultdict(int)
        }

        # Детектор лиц
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Для расчета FPS
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

        # Оптимизации
        self.last_face_processing_time = 0
        self.embedding_cache = {}
        self.cache_max_size = 100

        # Предзагрузка известных посетителей
        self._load_known_visitors()

        logger.info("🚀 БЫСТРАЯ система инициализирована")

    def _create_directories(self):
        """Создание папок для хранения фото"""
        os.makedirs(self.photos_dir, exist_ok=True)
        os.makedirs(os.path.join(self.photos_dir, self.current_session_dir), exist_ok=True)
        logger.info(f"📁 Созданы папки для фото: {self.photos_dir}")

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
                confirmed_count INTEGER DEFAULT 1,
                photo_path TEXT,
                quality_score REAL DEFAULT 1.0
            )
        ''')
        self.conn.commit()

    def _load_known_visitors(self):
        """Загрузка известных посетителей"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, face_embedding, photo_path FROM visitors")
        visitors = cursor.fetchall()

        self.known_visitors_cache.clear()
        for visitor_id, embedding_blob, photo_path in visitors:
            if embedding_blob:
                try:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    self.known_visitors_cache[visitor_id] = embedding
                except Exception as e:
                    logger.warning(f"Ошибка загрузки посетителя {visitor_id}: {e}")

        logger.info(f"📊 Загружено посетителей: {len(self.known_visitors_cache)}")

    def setup_rtsp_camera(self, rtsp_url):
        """Настройка RTSP подключения с минимальными задержками"""
        logger.info(f"📡 Подключение к камере: {rtsp_url}")

        # Параметры для минимальной задержки
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 25)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

        # Пропускаем первые кадры для очистки буфера
        for _ in range(3):
            cap.read()

        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                logger.info(f"✅ Камера подключена. Разрешение: {test_frame.shape[1]}x{test_frame.shape[0]}")
            else:
                logger.error("❌ Камера не передает данные")
        else:
            logger.error("❌ Не удалось подключиться к камере")

        return cap

    def resize_frame_fast(self, frame, max_width=1920):
        """Быстрое изменение размера с сохранением качества"""
        h, w = frame.shape[:2]
        if w <= max_width:
            return frame

        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)

        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def detect_faces_fast(self, frame):
        """Сверхбыстрая детекция лиц"""
        try:
            # Уменьшаем размер для быстрой обработки, но не слишком сильно
            small_frame = self.resize_frame_fast(frame, 960)  # 960px вместо 640
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # Оптимальные параметры для скорости и качества
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,  # Уменьшили для большей чувствительности
                minSize=(60, 60),  # Увеличили минимальный размер
                maxSize=(400, 400),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Масштабируем координаты обратно
            scale_x = frame.shape[1] / small_frame.shape[1]
            scale_y = frame.shape[0] / small_frame.shape[0]

            valid_faces = []
            for (x, y, w, h) in faces:
                scaled_bbox = (
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y)
                )

                # Базовая проверка размера
                if w * scale_x >= 60 and h * scale_y >= 60:
                    valid_faces.append(scaled_bbox)
                    logger.info(f"✅ Обнаружено лицо: {int(w * scale_x)}x{int(h * scale_y)} пикс")

            return valid_faces

        except Exception as e:
            logger.error(f"❌ Ошибка детекции: {e}")
            return []

    def get_embedding_fast(self, face_image):
        """Быстрое получение эмбеддинга с кэшированием"""
        try:
            # Кэширование по хэшу
            img_hash = hash(face_image.tobytes())
            if img_hash in self.embedding_cache:
                return self.embedding_cache[img_hash]

            # Быстрая обработка
            face_resized = cv2.resize(face_image, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            result = DeepFace.represent(
                face_rgb,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='skip',
                align=False
            )

            embedding = np.array(result[0]['embedding'], dtype=np.float32)

            # Обновление кэша
            if len(self.embedding_cache) >= self.cache_max_size:
                self.embedding_cache.clear()
            self.embedding_cache[img_hash] = embedding

            return embedding

        except Exception as e:
            logger.warning(f"❌ Ошибка эмбеддинга: {e}")
            return None

    def calculate_similarity(self, embedding1, embedding2):
        """Быстрое вычисление сходства"""
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0

            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return np.dot(embedding1, embedding2) / (norm1 * norm2)
        except:
            return 0.0

    def find_best_match(self, embedding):
        """Поиск лучшего совпадения"""
        best_match_id = None
        best_similarity = 0.0

        for visitor_id, known_embedding in self.known_visitors_cache.items():
            similarity = self.calculate_similarity(embedding, known_embedding)

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match_id = visitor_id

        return best_match_id, best_similarity

    def update_face_tracking_fast(self, faces, current_time):
        """Быстрое обновление трекинга"""
        active_tracks = {}

        with self.tracks_lock:
            # Очистка старых треков
            for track_id, track_info in list(self.face_tracks.items()):
                if current_time - track_info['last_seen'] > self.track_max_age:
                    del self.face_tracks[track_id]

            # Обновление треков
            for face_bbox in faces:
                x, y, w, h = face_bbox
                face_center = (x + w // 2, y + h // 2)

                best_track_id = None
                best_distance = float('inf')

                for track_id, track_info in self.face_tracks.items():
                    if current_time - track_info['last_seen'] > 2.0:
                        continue

                    last_center = track_info['last_center']
                    distance = math.sqrt((face_center[0] - last_center[0]) ** 2 +
                                         (face_center[1] - last_center[1]) ** 2)

                    max_distance = min(w, h) * 2.0

                    if distance < best_distance and distance < max_distance:
                        best_distance = distance
                        best_track_id = track_id

                if best_track_id is not None:
                    # Обновление существующего трека
                    self.face_tracks[best_track_id].update({
                        'last_seen': current_time,
                        'last_center': face_center,
                        'bbox': face_bbox,
                        'confirmed_count': self.face_tracks[best_track_id].get('confirmed_count', 0) + 1
                    })
                    active_tracks[best_track_id] = self.face_tracks[best_track_id]
                else:
                    # Создание нового трека
                    track_id = self.next_track_id
                    self.next_track_id += 1

                    self.face_tracks[track_id] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'last_center': face_center,
                        'bbox': face_bbox,
                        'confirmed_count': 1,
                        'status': 'detected'
                    }
                    active_tracks[track_id] = self.face_tracks[track_id]
                    logger.info(f"🆕 Создан трек {track_id}")

        return active_tracks

    def process_face_recognition(self, track_info, face_roi, track_id):
        """Обработка распознавания лица с логированием"""
        try:
            embedding = self.get_embedding_fast(face_roi)
            if embedding is None:
                return False

            visitor_id, similarity = self.find_best_match(embedding)

            if visitor_id is not None:
                # Известный посетитель
                track_info['visitor_id'] = visitor_id
                track_info['status'] = 'known'
                track_info['similarity'] = similarity

                cursor = self.conn.cursor()
                cursor.execute('''
                    UPDATE visitors 
                    SET last_seen = ?, visit_count = visit_count + 1, last_updated = ?
                    WHERE id = ?
                ''', (datetime.datetime.now(), datetime.datetime.now(), visitor_id))
                self.conn.commit()

                with self.stats_lock:
                    self.recognition_stats['known_visitors'] += 1
                    self.recognition_stats['faces_processed'] += 1

                logger.info(f"👤 Распознан известный посетитель ID:{visitor_id} (сходство: {similarity:.3f})")
                return True
            else:
                # Новый посетитель
                visitor_id = self.add_new_visitor(embedding, face_roi)
                if visitor_id:
                    track_info['visitor_id'] = visitor_id
                    track_info['status'] = 'new'

                    with self.stats_lock:
                        self.recognition_stats['new_visitors'] += 1
                        self.recognition_stats['faces_processed'] += 1

                    logger.info(f"🆕 Добавлен новый посетитель ID:{visitor_id}")
                    return True

            return False

        except Exception as e:
            logger.error(f"❌ Ошибка распознавания: {e}")
            return False

    def add_new_visitor(self, embedding, face_image):
        """Добавление нового посетителя"""
        try:
            cursor = self.conn.cursor()

            # Сохранение фото
            filename = f"visitor_{int(time.time())}.jpg"
            filepath = os.path.join(self.photos_dir, self.current_session_dir, filename)
            cv2.imwrite(filepath, face_image)

            cursor.execute('''
                INSERT INTO visitors 
                (face_embedding, first_seen, last_seen, last_updated, photo_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (embedding.tobytes(), datetime.datetime.now(), datetime.datetime.now(),
                  datetime.datetime.now(), filepath))

            new_visitor_id = cursor.lastrowid
            self.conn.commit()

            # Обновление кэша
            self.known_visitors_cache[new_visitor_id] = embedding

            return new_visitor_id

        except Exception as e:
            logger.error(f"❌ Ошибка добавления посетителя: {e}")
            return None

    def process_frame_realtime(self, frame):
        """Обработка кадра в реальном времени без задержек"""
        current_time = time.time()

        # Обновление FPS каждую секунду
        self.fps_frame_count += 1
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_frame_count / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
            self.fps_frame_count = 0

        # Детекция лиц
        faces = self.detect_faces_fast(frame)

        # Трекинг
        active_tracks = self.update_face_tracking_fast(faces, current_time)

        # Отрисовка и обработка
        processed_frame = frame.copy()
        detected_count = len(active_tracks)
        processed_count = 0

        for track_id, track_info in active_tracks.items():
            x, y, w, h = track_info['bbox']

            # Определение цвета и статуса
            status = track_info.get('status', 'detected')
            color = self.COLORS[status]

            # Отрисовка bounding box
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 3)

            # Подготовка текста
            if status == 'known':
                visitor_id = track_info.get('visitor_id', '?')
                similarity = track_info.get('similarity', 0)
                label = f"KNOWN ID:{visitor_id} ({similarity:.2f})"
            elif status == 'new':
                visitor_id = track_info.get('visitor_id', '?')
                label = f"NEW ID:{visitor_id}"
            else:
                conf_count = track_info.get('confirmed_count', 1)
                label = f"TRACK {track_id} ({conf_count})"

            # Отрисовка текста с фоном для читаемости
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(processed_frame, (x, y - text_size[1] - 10),
                          (x + text_size[0], y), color, -1)
            cv2.putText(processed_frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Обработка распознавания для стабильных треков
            if (track_info['confirmed_count'] >= 2 and
                    current_time - track_info.get('last_processed', 0) > self.processing_interval and
                    track_info.get('status') in [None, 'detected']):

                face_roi = frame[y:y + h, x:x + w]
                if self.process_face_recognition(track_info, face_roi, track_id):
                    processed_count += 1

                track_info['last_processed'] = current_time

        # Обновление статистики
        with self.stats_lock:
            self.recognition_stats['total_detections'] += len(faces)
            self.recognition_stats['valid_detections'] += detected_count
            self.recognition_stats['frames_processed'] += 1

        return processed_frame, detected_count, processed_count

    def start_analysis(self, rtsp_url):
        """Запуск анализа с большим окном и минимальными задержками"""
        logger.info("🚀 Запуск БЫСТРОЙ версии...")

        cap = self.setup_rtsp_camera(rtsp_url)
        if not cap.isOpened():
            return

        # Создаем большое окно
        window_name = 'Trassir Analytics - FAST MODE'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 1200)  # Большое окно

        logger.info("✅ Анализ запущен!")

        try:
            while True:
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    logger.warning("📡 Потеряно соединение...")
                    time.sleep(0.5)
                    continue

                # Обработка кадра
                processed_frame, detected, processed = self.process_frame_realtime(frame)

                # Подготовка к отображению
                display_frame = self.resize_frame_fast(processed_frame, 1600)

                # Статистика на экране (крупный текст)
                stats_text = [
                    f"FAST MODE - REAL TIME",
                    f"FPS: {self.current_fps:.1f}",
                    f"Active Faces: {detected}",
                    f"Total Tracks: {len(self.face_tracks)}",
                    f"Faces Processed: {processed}",
                    f"Known: {self.recognition_stats['known_visitors']}",
                    f"New: {self.recognition_stats['new_visitors']}",
                    f"Press Q to quit"
                ]

                # Отрисовка статистики с фоном
                for i, text in enumerate(stats_text):
                    y_position = 40 + i * 35
                    cv2.rectangle(display_frame, (10, y_position - 30),
                                  (600, y_position + 5), (0, 0, 0), -1)
                    cv2.putText(display_frame, text, (15, y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow(window_name, display_frame)

                # Минимальная задержка для отзывчивости
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("⏹️ Остановка по Ctrl+C...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.conn.close()

            # Финальная статистика
            logger.info(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
            logger.info(f"   Средний FPS: {self.current_fps:.1f}")
            logger.info(f"   Обработано кадров: {self.recognition_stats['frames_processed']}")
            logger.info(f"   Обнаружено лиц: {self.recognition_stats['valid_detections']}")
            logger.info(f"   Обработано лиц: {self.recognition_stats['faces_processed']}")
            logger.info(f"   Известных: {self.recognition_stats['known_visitors']}")
            logger.info(f"   Новых: {self.recognition_stats['new_visitors']}")
            logger.info("✅ Анализ завершен")


def main():
    """Основная функция"""
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    counter = FastTrassirCounter(
        processing_interval=1.5,
        similarity_threshold=0.65,
        tracking_threshold=0.50
    )

    try:
        counter.start_analysis(RTSP_URL)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()