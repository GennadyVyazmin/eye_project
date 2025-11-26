# video_analytics_trassir_optimized.py
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedTrassirCounter:
    def __init__(self, processing_interval=2.0, similarity_threshold=0.65, tracking_threshold=0.50):
        """
        ОПТИМИЗИРОВАННАЯ версия для снижения нагрузки на CPU
        """
        self.conn = sqlite3.connect('visitors_trassir_optimized.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval  # Увеличили интервал обработки
        self.similarity_threshold = similarity_threshold
        self.tracking_threshold = tracking_threshold

        # Мьютексы для потокобезопасности
        self.stats_lock = Lock()
        self.tracks_lock = Lock()
        self.gallery_lock = Lock()

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
        self.photos_dir = "visitor_photos_optimized"
        self.current_session_dir = "current_session"
        self._create_directories()

        # Трекинг состояния
        self.last_processing_time = 0
        self.known_visitors_cache = {}
        self.frame_count = 0

        # Система трекинга лиц
        self.face_tracks = {}
        self.next_track_id = 1
        self.track_max_age = 8.0

        # Галерея текущих посетителей с автоочисткой
        self.current_visitors_gallery = {}
        self.gallery_max_size = 6  # Уменьшили размер галереи

        # ОПТИМИЗАЦИЯ: Упрощенные фильтры для скорости
        self.false_positive_filter = {
            'min_face_width': 50,  # Увеличили минимальный размер для скорости
            'min_face_height': 50,
            'min_brightness': 20,
            'max_brightness': 240,
            'required_confirmations': 2,  # Требуем больше подтверждений
        }

        # Статистика
        self.recognition_stats = {
            'total_detections': 0,
            'valid_detections': 0,
            'rejected_detections': 0,
            'new_visitors': 0,
            'known_visitors': 0,
            'frames_processed': 0,
            'quality_rejections': defaultdict(int)
        }
        self.last_log_time = time.time()

        # Очередь для обработки - ОГРАНИЧИЛИ размер
        self.frame_queue = Queue(maxsize=1)  # Всего 1 кадр в очереди
        self.results_queue = Queue()

        # ОПТИМИЗАЦИЯ: Загружаем каскады один раз
        logger.info("🔧 Загрузка детекторов лиц...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.alt_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )

        # ОПТИМИЗАЦИЯ: Кэш для эмбеддингов
        self.embedding_cache = {}
        self.cache_max_size = 50

        # Предзагрузка известных посетителей
        self._load_known_visitors()

        # Поток для обработки
        self.processing_thread = None
        self.stop_processing = False

        # Для расчета FPS
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

        # ОПТИМИЗАЦИЯ: Счетчик для ограничения обработки
        self.frame_skip_counter = 0
        self.frame_skip_interval = 2  # Обрабатываем каждый 3-й кадр

        logger.info("🎯 ОПТИМИЗИРОВАННАЯ система инициализирована")

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
        """Настройка RTSP подключения с оптимизацией"""
        logger.info(f"📡 Подключение к камере: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)

        # ОПТИМИЗАЦИЯ: Упрощенные настройки камеры
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 10)  # Уменьшили FPS камеры

        # Пропускаем первые кадры
        for _ in range(5):
            cap.read()

        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                # ОПТИМИЗАЦИЯ: Уменьшаем разрешение для обработки
                h, w = test_frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    new_w = 1280
                    new_h = int(h * scale)
                    logger.info(f"📐 Будет использовано разрешение: {new_w}x{new_h} (оригинал: {w}x{h})")

                logger.info(f"✅ Камера подключена")
            else:
                logger.error("❌ Камера не передает данные")
        else:
            logger.error("❌ Не удалось подключиться к камере")

        return cap

    def resize_frame_optimized(self, frame, max_width=1280):
        """Быстрое изменение размера кадра"""
        h, w = frame.shape[:2]
        if w <= max_width:
            return frame

        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)

        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def analyze_face_quality_fast(self, face_image, bbox):
        """БЫСТРЫЙ анализ качества лица"""
        try:
            x, y, w, h = bbox

            # ТОЛЬКО КРИТИЧЕСКИЕ ПРОВЕРКИ
            if w < self.false_positive_filter['min_face_width']:
                return False, "small_width"
            if h < self.false_positive_filter['min_face_height']:
                return False, "small_height"

            # Быстрая проверка яркости
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_face)

            if brightness < self.false_positive_filter['min_brightness']:
                return False, "dark"
            if brightness > self.false_positive_filter['max_brightness']:
                return False, "bright"

            return True, "valid"

        except Exception as e:
            return False, f"error: {e}"

    def detect_faces_fast(self, frame):
        """БЫСТРАЯ детекция лиц"""
        # ОПТИМИЗАЦИЯ: Уменьшаем размер для детекции
        small_frame = self.resize_frame_optimized(frame, 640)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # ОПТИМИЗАЦИЯ: Только один каскад с оптимальными параметрами
        scale_factor = 1.1
        min_neighbors = 4
        min_size = (50, 50)
        max_size = (300, 300)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            maxSize=max_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Масштабируем координаты обратно к оригинальному размеру
        scale_x = frame.shape[1] / small_frame.shape[1]
        scale_y = frame.shape[0] / small_frame.shape[0]

        scaled_faces = []
        for (x, y, w, h) in faces:
            scaled_faces.append((
                int(x * scale_x),
                int(y * scale_y),
                int(w * scale_x),
                int(h * scale_y)
            ))

        # Быстрая фильтрация
        valid_faces = []
        for bbox in scaled_faces:
            x, y, w, h = bbox
            face_roi = frame[y:y + h, x:x + w]

            is_valid, reason = self.analyze_face_quality_fast(face_roi, bbox)
            if is_valid:
                valid_faces.append(bbox)
            else:
                with self.stats_lock:
                    self.recognition_stats['quality_rejections'][reason] += 1
                    self.recognition_stats['rejected_detections'] += 1

        logger.debug(f"🔍 Найдено лиц: {len(valid_faces)}")
        return valid_faces

    def get_fast_embedding_optimized(self, face_image):
        """ОПТИМИЗИРОВАННОЕ получение эмбеддинга"""
        try:
            # Проверка кэша (по хэшу изображения)
            img_hash = hash(face_image.tobytes())
            if img_hash in self.embedding_cache:
                return self.embedding_cache[img_hash]

            # ОПТИМИЗАЦИЯ: Фиксированный размер для скорости
            face_resized = cv2.resize(face_image, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # ОПТИМИЗАЦИЯ: Быстрая модель
            result = DeepFace.represent(
                face_rgb,
                model_name='Facenet',  # Можно попробовать 'OpenFace' для скорости
                enforce_detection=False,
                detector_backend='skip',  # Пропускаем детекцию, т.к. лицо уже выделено
                align=False
            )

            embedding = np.array(result[0]['embedding'], dtype=np.float32)

            # Сохраняем в кэш
            if len(self.embedding_cache) >= self.cache_max_size:
                self.embedding_cache.clear()
            self.embedding_cache[img_hash] = embedding

            return embedding

        except Exception as e:
            logger.warning(f"❌ Ошибка получения эмбеддинга: {e}")
            return None

    def calculate_similarity(self, embedding1, embedding2):
        """Вычисление косинусного сходства"""
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0

            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            return 0.0

    def find_best_match(self, embedding):
        """Быстрый поиск совпадения"""
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
            # Быстрая очистка старых треков
            for track_id, track_info in list(self.face_tracks.items()):
                if current_time - track_info['last_seen'] > self.track_max_age:
                    del self.face_tracks[track_id]

            # Быстрое обновление треков
            for face_bbox in faces:
                x, y, w, h = face_bbox
                face_center = (x + w // 2, y + h // 2)

                best_track_id = None
                best_distance = float('inf')

                for track_id, track_info in self.face_tracks.items():
                    if current_time - track_info['last_seen'] > 1.0:
                        continue

                    last_center = track_info['last_center']
                    distance = math.sqrt((face_center[0] - last_center[0]) ** 2 +
                                         (face_center[1] - last_center[1]) ** 2)

                    max_distance = min(w, h) * 1.5

                    if distance < best_distance and distance < max_distance:
                        best_distance = distance
                        best_track_id = track_id

                if best_track_id is not None:
                    # Обновление трека
                    self.face_tracks[best_track_id].update({
                        'last_seen': current_time,
                        'last_center': face_center,
                        'bbox': face_bbox,
                        'confirmed_count': self.face_tracks[best_track_id].get('confirmed_count', 0) + 1
                    })
                    active_tracks[best_track_id] = self.face_tracks[best_track_id]
                else:
                    # Новый трек
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

        return active_tracks

    def process_frame_optimized(self, frame):
        """ОПТИМИЗИРОВАННАЯ обработка кадра"""
        current_time = time.time()

        # ОПТИМИЗАЦИЯ: Пропускаем кадры для снижения нагрузки
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.frame_skip_interval != 0:
            return frame, 0, 0

        # Обновление FPS
        self.fps_frame_count += 1
        if current_time - self.fps_start_time >= 2.0:  # Раз в 2 секунды
            self.current_fps = self.fps_frame_count / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
            self.fps_frame_count = 0

        # БЫСТРАЯ детекция лиц
        faces = self.detect_faces_fast(frame)

        # Быстрое обновление трекинга
        active_tracks = self.update_face_tracking_fast(faces, current_time)

        # Отрисовка результатов
        processed_frame = frame.copy()
        detected_count = 0
        processed_count = 0

        for track_id, track_info in active_tracks.items():
            x, y, w, h = track_info['bbox']

            # Определение статуса
            if track_info.get('visitor_id'):
                status = 'known' if track_info.get('status') == 'known' else 'new'
            elif track_info['confirmed_count'] >= self.false_positive_filter['required_confirmations']:
                status = 'tracking'
            else:
                status = 'detected'

            color = self.COLORS[status]

            # Отрисовка
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
            label = f"ID:{track_id}"
            cv2.putText(processed_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            detected_count += 1

            # ОПТИМИЗАЦИЯ: Распознавание только для стабильных треков
            if (track_info['confirmed_count'] >= 3 and
                    current_time - track_info.get('last_processed', 0) > self.processing_interval):

                try:
                    face_roi = frame[y:y + h, x:x + w]
                    embedding = self.get_fast_embedding_optimized(face_roi)

                    if embedding is not None:
                        visitor_id, similarity = self.find_best_match(embedding)
                        if visitor_id:
                            track_info['visitor_id'] = visitor_id
                            track_info['status'] = 'known'
                            processed_count += 1
                            logger.info(f"👤 Распознан посетитель {visitor_id} (сходство: {similarity:.3f})")

                    track_info['last_processed'] = current_time

                except Exception as e:
                    logger.debug(f"Ошибка обработки лица: {e}")

        with self.stats_lock:
            self.recognition_stats['total_detections'] += len(faces)
            self.recognition_stats['valid_detections'] += detected_count
            self.recognition_stats['frames_processed'] += 1

        return processed_frame, detected_count, processed_count

    def start_analysis_optimized(self, rtsp_url):
        """Запуск ОПТИМИЗИРОВАННОГО анализа"""
        logger.info("🚀 Запуск ОПТИМИЗИРОВАННОЙ версии...")

        cap = self.setup_rtsp_camera(rtsp_url)
        if not cap.isOpened():
            return

        logger.info("✅ Анализ запущен в ОПТИМИЗИРОВАННОМ режиме!")

        window_name = 'Trassir Analytics - OPTIMIZED'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    logger.warning("📡 Потеряно соединение...")
                    time.sleep(1)
                    continue

                # ОПТИМИЗАЦИЯ: Обработка в основном потоке
                processed_frame, detected, processed = self.process_frame_optimized(frame)

                # ОПТИМИЗАЦИЯ: Уменьшаем для отображения
                display_frame = self.resize_frame_optimized(processed_frame, 1280)

                # Статистика на экране
                stats_text = [
                    f"OPTIMIZED MODE - CPU SAVER",
                    f"FPS: {self.current_fps:.1f}",
                    f"Detected: {detected}",
                    f"Tracks: {len(self.face_tracks)}",
                    f"Frame skip: {self.frame_skip_interval}",
                    f"Press 'q' to quit"
                ]

                # Простой оверлей
                for i, text in enumerate(stats_text):
                    cv2.putText(display_frame, text, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(display_frame, text, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow(window_name, display_frame)

                # ОПТИМИЗАЦИЯ: Задержка для снижения нагрузки
                processing_time = time.time() - start_time
                delay = max(1, int(30 - processing_time * 1000))  # Целевой FPS ~30

                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("⏹️ Остановка...")
        finally:
            self.stop_processing = True
            cap.release()
            cv2.destroyAllWindows()
            self.conn.close()

            logger.info(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
            logger.info(f"   Обработано кадров: {self.recognition_stats['frames_processed']}")
            logger.info(f"   Обнаружено лиц: {self.recognition_stats['valid_detections']}")
            logger.info(f"   Средний FPS: {self.current_fps:.1f}")
            logger.info("✅ Анализ завершен")


def main():
    """Основная функция"""
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    # ОПТИМИЗАЦИЯ: Увеличиваем интервалы обработки
    counter = OptimizedTrassirCounter(
        processing_interval=3.0,  # Увеличили интервал обработки
        similarity_threshold=0.60,
        tracking_threshold=0.45
    )

    try:
        counter.start_analysis_optimized(RTSP_URL)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()