# video_analytics_trassir_enhanced.py
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

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedTrassirCounter:
    def __init__(self, processing_interval=1.0, similarity_threshold=0.55, tracking_threshold=0.45):
        """
        Улучшенная версия с MediaPipe и фильтрацией ложных срабатываний
        """
        self.conn = sqlite3.connect('visitors_trassir_enhanced.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
        self.similarity_threshold = similarity_threshold
        self.tracking_threshold = tracking_threshold

        # Параметры фильтрации
        self.min_face_size = 80
        self.max_face_size = 300
        self.min_confidence = 0.7

        # Цвета для индикации статусов
        self.COLORS = {
            'detected': (0, 255, 0),  # Зеленый - лицо обнаружено
            'tracking': (255, 255, 0),  # Желтый - создан трек
            'known': (0, 255, 255),  # Голубой - известный пользователь
            'new': (0, 0, 255),  # Красный - новый пользователь в БД
            'analyzing': (255, 165, 0),  # Оранжевый - анализ в процессе
            'filtered': (128, 128, 128)  # Серый - отфильтровано
        }

        # Папки для хранения фото
        self.photos_dir = "visitor_photos_enhanced"
        self.current_session_dir = "current_session"
        self._create_directories()

        # Трекинг состояния
        self.last_processing_time = 0
        self.known_visitors_cache = {}
        self.frame_count = 0

        # Система трекинга лиц
        self.face_tracks = {}
        self.next_track_id = 1
        self.track_max_age = 5.0

        # Галерея текущих посетителей
        self.current_visitors_gallery = {}
        self.gallery_max_size = 8
        self.photo_size = (120, 160)

        # Статистика
        self.recognition_stats = {
            'total_detections': 0,
            'new_visitors': 0,
            'known_visitors': 0,
            'frames_processed': 0,
            'filtered_detections': 0
        }
        self.last_log_time = time.time()

        # Очередь для обработки
        self.frame_queue = Queue(maxsize=1)
        self.results_queue = Queue()

        # Инициализация детекторов
        self.setup_face_detection_models()

        # Предзагрузка известных посетителей
        self._load_known_visitors()

        # Поток для обработки
        self.processing_thread = None
        self.stop_processing = False

        # Для расчета FPS
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

        logger.info("🎯 Улучшенная система инициализирована с MediaPipe")

    def setup_face_detection_models(self):
        """Инициализация моделей детекции лиц"""
        try:
            # Инициализация MediaPipe
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 для ближних, 1 для дальних лиц
                min_detection_confidence=0.5
            )
            self.use_mediapipe = True
            logger.info("✅ MediaPipe инициализирован для детекции лиц")
        except ImportError as e:
            self.use_mediapipe = False
            logger.warning(f"❌ MediaPipe не доступен: {e}. Используем OpenCV")

        # Резервные каскады OpenCV
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.alt_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )

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
                photo_path TEXT
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

    def detect_faces_mediapipe(self, frame):
        """Детекция лиц с использованием MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Увеличиваем bounding box немного для лучшего захвата
                x = max(0, x - 15)
                y = max(0, y - 15)
                width = min(w - x, width + 30)
                height = min(h - y, height + 30)

                confidence = detection.score[0]

                # Фильтрация по уверенности и размеру
                if (confidence >= self.min_confidence and
                        width >= self.min_face_size and height >= self.min_face_size and
                        width <= self.max_face_size and height <= self.max_face_size):

                    # Дополнительная проверка на валидность
                    if self.is_valid_face_region(frame, x, y, width, height):
                        faces.append((x, y, width, height))
                        logger.debug(f"✅ MediaPipe обнаружено лицо с уверенностью {confidence:.3f}")

        return faces

    def detect_faces_opencv(self, frame):
        """Резервная детекция лиц с OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Основной каскад с улучшенными параметрами
        faces1 = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=7,
            minSize=(self.min_face_size, self.min_face_size),
            maxSize=(self.max_face_size, self.max_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Фильтрация ложных срабатываний
        valid_faces = []
        for (x, y, w, h) in faces1:
            # Проверка соотношения сторон
            aspect_ratio = w / h
            if 0.7 < aspect_ratio < 1.8:
                # Проверка текстуры и характеристик
                roi = gray[y:y + h, x:x + w]
                if self.is_likely_face(roi) and self.is_valid_face_region(frame, x, y, w, h):
                    valid_faces.append((x, y, w, h))

        return valid_faces

    def is_valid_face_region(self, frame, x, y, w, h):
        """Проверка валидности региона лица"""
        h_total, w_total = frame.shape[:2]

        # Проверка выхода за границы
        if x < 0 or y < 0 or x + w > w_total or y + h > h_total:
            return False

        # Проверка размера (относительно размера кадра)
        if w < w_total * 0.05 or h < h_total * 0.05:  # Слишком маленький
            return False
        if w > w_total * 0.4 or h > h_total * 0.4:  # Слишком большой
            return False

        return True

    def is_likely_face(self, face_roi):
        """Проверка, что регион вероятнее всего является лицом"""
        if face_roi.size == 0:
            return False

        # Проверка вариации интенсивности (лица обычно имеют высокий контраст)
        std_dev = np.std(face_roi)
        if std_dev < 20:  # Слишком однородная текстура - вероятно не лицо
            return False

        # Проверка гистограммы
        hist = cv2.calcHist([face_roi], [0], None, [8], [0, 256])
        hist = hist.flatten()
        if hist.sum() > 0:
            hist = hist / hist.sum()

        # Поиск пиков в гистограмме
        peak_count = 0
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > 0.1:
                peak_count += 1

        return peak_count >= 1

    def validate_human_features(self, face_image):
        """Проверка что обнаруженный объект имеет характеристики человека"""
        if face_image.size == 0:
            return False

        h, w = face_image.shape[:2]

        # Проверка размера
        if w < 50 or h < 50 or w > 400 or h > 400:
            return False

        try:
            # Проверка цветового распределения (кожа человека)
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)

            # Маска для цветов кожи
            skin_lower = np.array([0, 20, 70], dtype=np.uint8)
            skin_upper = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)

            # Процент пикселей кожи
            skin_ratio = np.sum(skin_mask > 0) / (w * h)

            # Для лиц обычно 15-50% пикселей соответствуют цвету кожи
            return 0.1 < skin_ratio < 0.7
        except:
            return True  # Если не удалось проверить, даем шанс

    def detect_faces_enhanced(self, frame):
        """Улучшенная детекция лиц с приоритетом MediaPipe"""
        if self.use_mediapipe:
            faces = self.detect_faces_mediapipe(frame)
            if faces:
                logger.info(f"🔍 MediaPipe: {len(faces)} лиц")
                return faces

        # Резервный вариант с OpenCV
        faces = self.detect_faces_opencv(frame)
        logger.info(f"🔍 OpenCV: {len(faces)} лиц")

        return faces

    def get_fast_embedding(self, face_image):
        """Быстрое получение эмбеддинга"""
        try:
            # Минимальная предобработка для скорости
            face_resized = cv2.resize(face_image, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            result = DeepFace.represent(
                face_rgb,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='opencv',
                align=False
            )

            embedding = np.array(result[0]['embedding'], dtype=np.float32)

            if np.all(embedding == 0) or np.linalg.norm(embedding) < 0.1:
                return None

            return embedding

        except Exception as e:
            logger.warning(f"Ошибка получения эмбеддинга: {e}")
            return None

    def calculate_similarity(self, embedding1, embedding2):
        """Расчет схожести лиц"""
        if embedding1 is None or embedding2 is None:
            return 0.0

        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            emb1_norm = embedding1 / norm1
            emb2_norm = embedding2 / norm2

            similarity = float(np.dot(emb1_norm, emb2_norm))
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.debug(f"Ошибка расчета схожести: {e}")
            return 0.0

    def find_best_match(self, embedding):
        """Поиск лучшего совпадения"""
        if embedding is None:
            return None, 0.0

        best_match_id = None
        best_similarity = 0.0

        for visitor_id, known_embedding in self.known_visitors_cache.items():
            similarity = self.calculate_similarity(embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = visitor_id

        return best_match_id, best_similarity

    def update_face_tracking(self, current_faces, timestamp):
        """Обновление трекинга лиц между кадрами"""
        updated_faces = []

        for face_data in current_faces:
            embedding = face_data['embedding']
            coords = face_data['coords']
            face_image = face_data['face_image']

            best_track_id = None
            best_similarity = 0.0

            # Удаляем старые треки
            for track_id in list(self.face_tracks.keys()):
                if timestamp - self.face_tracks[track_id]['last_seen'] > self.track_max_age:
                    logger.debug(f"🗑️ Удален старый трек {track_id}")
                    del self.face_tracks[track_id]

            # Ищем совпадение с существующими треками
            for track_id, track_data in self.face_tracks.items():
                similarity = self.calculate_similarity(embedding, track_data['embedding'])
                if similarity > best_similarity and similarity > self.tracking_threshold:
                    best_similarity = similarity
                    best_track_id = track_id

            if best_track_id is not None:
                # Обновляем существующий трек
                self.face_tracks[best_track_id].update({
                    'embedding': embedding,
                    'last_seen': timestamp,
                    'coords': coords,
                    'face_image': face_image
                })
                face_data['track_id'] = best_track_id
                face_data['visitor_id'] = self.face_tracks[best_track_id].get('visitor_id')
                face_data['status'] = 'tracking'
                logger.debug(f"🔄 Обновлен трек {best_track_id}, схожесть: {best_similarity:.3f}")
            else:
                # Создаем новый трек
                track_id = self.next_track_id
                self.next_track_id += 1
                self.face_tracks[track_id] = {
                    'embedding': embedding,
                    'last_seen': timestamp,
                    'coords': coords,
                    'face_image': face_image,
                    'visitor_id': None,
                    'created_at': timestamp
                }
                face_data['track_id'] = track_id
                face_data['status'] = 'tracking'
                logger.info(f"🎯 Создан новый трек {track_id}")

            updated_faces.append(face_data)

        return updated_faces

    def confirm_visitor_identity(self, track_id, face_data):
        """Подтверждение идентичности посетителя"""
        track_data = self.face_tracks.get(track_id)
        if not track_data:
            return None

        embedding = face_data['embedding']
        face_image = face_data['face_image']
        timestamp = time.time()

        # Если уже есть visitor_id, обновляем его
        if track_data['visitor_id']:
            visitor_id = track_data['visitor_id']
            self.recognition_stats['known_visitors'] += 1
            face_data['status'] = 'known'

            # Обновляем галерею
            self.update_visitor_gallery(visitor_id, face_image, 'known')

            logger.debug(f"♻️  Подтвержден известный посетитель {visitor_id}")
            return visitor_id

        # Ищем лучшего кандидата среди известных
        visitor_id, similarity = self.find_best_match(embedding)

        if similarity > self.similarity_threshold:
            # Подтверждаем существующего посетителя
            track_data['visitor_id'] = visitor_id
            track_data['confirmed_at'] = timestamp
            self.recognition_stats['known_visitors'] += 1
            face_data['status'] = 'known'

            # Обновляем галерею
            self.update_visitor_gallery(visitor_id, face_image, 'known')

            logger.info(f"👤 ОПОЗНАН известный посетитель {visitor_id}, схожесть: {similarity:.3f}")
            return visitor_id
        else:
            # Ждем подтверждения для нового посетителя
            track_duration = timestamp - track_data['created_at']
            if track_duration > 2.0:  # Минимум 2 секунды трекинга
                new_visitor_id = self._create_new_visitor(embedding, face_image, track_id)
                if new_visitor_id:
                    self.recognition_stats['new_visitors'] += 1
                    face_data['status'] = 'new'

                    # Сохраняем фото и обновляем галерею
                    self.update_visitor_gallery(new_visitor_id, face_image, 'new')

                    logger.info(f"🆕 СОЗДАН новый посетитель {new_visitor_id}, схожесть с известными: {similarity:.3f}")
                return new_visitor_id
            else:
                face_data['status'] = 'analyzing'
                logger.debug(f"⏳ Трек {track_id} ожидает подтверждения ({track_duration:.1f}s)")

        return None

    def _create_new_visitor(self, embedding, face_image, track_id):
        """Создание нового посетителя"""
        cursor = self.conn.cursor()
        now = datetime.datetime.now()

        visitor_id = None
        try:
            # Сначала сохраняем фото чтобы получить путь
            photo_path = self.save_visitor_photo(face_image, "temp")

            # Создаем запись в базе
            embedding_blob = embedding.astype(np.float32).tobytes()
            cursor.execute(
                """INSERT INTO visitors (face_embedding, first_seen, last_seen, 
                   visit_count, last_updated, confirmed_count, photo_path) 
                   VALUES (?, ?, ?, 1, ?, 1, ?)""",
                (embedding_blob, now, now, now, photo_path)
            )
            visitor_id = cursor.lastrowid

            # Обновляем путь к фото с правильным ID
            final_photo_path = self.save_visitor_photo(face_image, visitor_id)
            cursor.execute(
                "UPDATE visitors SET photo_path = ? WHERE id = ?",
                (final_photo_path, visitor_id)
            )

            self.known_visitors_cache[visitor_id] = embedding
            self.conn.commit()

            # Удаляем временное фото
            if os.path.exists(photo_path):
                os.remove(photo_path)

        except Exception as e:
            logger.error(f"❌ Ошибка создания посетителя: {e}")
            self.conn.rollback()
            return None

        if track_id in self.face_tracks:
            self.face_tracks[track_id]['visitor_id'] = visitor_id

        return visitor_id

    def save_visitor_photo(self, face_image, visitor_id):
        """Сохранение фото посетителя"""
        try:
            photo_clean = face_image.copy()

            height, width = photo_clean.shape[:2]
            if width < 200:
                scale = 200 / width
                new_width = 200
                new_height = int(height * scale)
                photo_clean = cv2.resize(photo_clean, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            filename = f"visitor_{visitor_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(self.photos_dir, filename)
            session_filepath = os.path.join(self.photos_dir, self.current_session_dir, filename)

            cv2.imwrite(filepath, photo_clean)
            cv2.imwrite(session_filepath, photo_clean)

            logger.info(f"📸 Сохранено фото посетителя {visitor_id}")
            return filepath

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения фото: {e}")
            return ""

    def update_visitor_gallery(self, visitor_id, face_image, status):
        """Обновление галереи текущих посетителей"""
        try:
            gallery_photo = face_image.copy()

            border_color = self.COLORS.get(status, (255, 255, 255))
            gallery_photo = cv2.copyMakeBorder(
                gallery_photo, 5, 25, 5, 5, cv2.BORDER_CONSTANT, value=border_color
            )

            status_text = self.get_status_text(status)
            cv2.putText(gallery_photo, f"ID: {visitor_id}", (10, gallery_photo.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)

            gallery_photo = cv2.resize(gallery_photo, self.photo_size, interpolation=cv2.INTER_AREA)

            self.current_visitors_gallery[visitor_id] = {
                'photo': gallery_photo,
                'last_seen': time.time(),
                'status': status
            }

            if len(self.current_visitors_gallery) > self.gallery_max_size:
                oldest_visitor = min(self.current_visitors_gallery.keys(),
                                     key=lambda x: self.current_visitors_gallery[x]['last_seen'])
                del self.current_visitors_gallery[oldest_visitor]

        except Exception as e:
            logger.error(f"Ошибка обновления галереи: {e}")

    def get_status_text(self, status):
        """Получение текста по статусу"""
        status_texts = {
            'detected': 'DETECTED',
            'tracking': 'TRACKING',
            'analyzing': 'ANALYZING',
            'known': 'KNOWN',
            'new': 'NEW USER',
            'filtered': 'FILTERED'
        }
        return status_texts.get(status, 'UNKNOWN')

    def get_color_by_status(self, status):
        """Получение цвета по статусу"""
        return self.COLORS.get(status, (255, 255, 255))

    def create_gallery_display(self, main_frame):
        """Создание галереи посетителей"""
        try:
            main_height, main_width = main_frame.shape[:2]

            gallery_width = 300
            gallery_panel = np.zeros((main_height, gallery_width, 3), dtype=np.uint8)

            cv2.putText(gallery_panel, "CURRENT VISITORS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(gallery_panel, f"Total: {len(self.current_visitors_gallery)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if self.current_visitors_gallery:
                visitor_ids = sorted(self.current_visitors_gallery.keys())
                photos_per_column = 4
                photo_width, photo_height = self.photo_size
                margin = 10

                for i, visitor_id in enumerate(visitor_ids):
                    if i >= self.gallery_max_size:
                        break

                    visitor_data = self.current_visitors_gallery[visitor_id]
                    row = i % photos_per_column
                    col = i // photos_per_column

                    x = margin + col * (photo_width + margin)
                    y = 80 + row * (photo_height + margin)

                    if y + photo_height < main_height and x + photo_width < gallery_width:
                        gallery_panel[y:y + photo_height, x:x + photo_width] = visitor_data['photo']

                        status_color = self.COLORS.get(visitor_data['status'], (255, 255, 255))
                        cv2.circle(gallery_panel, (x + 10, y + 10), 5, status_color, -1)
            else:
                cv2.putText(gallery_panel, "No visitors", (50, main_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
                cv2.putText(gallery_panel, "in frame", (60, main_height // 2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)

            combined_frame = np.hstack([main_frame, gallery_panel])
            return combined_frame

        except Exception as e:
            logger.error(f"Ошибка создания галереи: {e}")
            return main_frame

    def resize_frame_for_display(self, frame, target_width=1280):
        """Изменение размера кадра для отображения"""
        height, width = frame.shape[:2]

        if width <= target_width:
            return frame

        ratio = target_width / width
        new_width = target_width
        new_height = int(height * ratio)

        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        return resized_frame

    def log_recognition_stats(self):
        """Логирование статистики распознавания"""
        current_time = time.time()
        if current_time - self.last_log_time >= 3.0:  # Каждые 3 секунды
            logger.info(f"📊 СТАТИСТИКА: Всего в базе: {len(self.known_visitors_cache)}, "
                        f"Активных треков: {len(self.face_tracks)}, "
                        f"В галерее: {len(self.current_visitors_gallery)}, "
                        f"Новых за сессию: {self.recognition_stats['new_visitors']}, "
                        f"Известных: {self.recognition_stats['known_visitors']}, "
                        f"Отфильтровано: {self.recognition_stats['filtered_detections']}")
            self.last_log_time = current_time

    def start_processing_thread(self):
        """Запуск фонового потока обработки"""
        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        logger.info("Фоновый поток обработки запущен")

    def _processing_worker(self):
        """Фоновая обработка кадров"""
        while not self.stop_processing:
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                frame, frame_time = frame_data

                result = self._process_frame_heavy(frame)
                self.results_queue.put((result, frame_time))

                self.frame_queue.task_done()

            except:
                continue

    def _process_frame_heavy(self, frame):
        """Тяжелые операции обработки с улучшенной фильтрацией"""
        try:
            # Детекция лиц с улучшенной фильтрацией
            faces = self.detect_faces_enhanced(frame)

            processed_faces = []
            filtered_count = 0

            if len(faces) > 0:
                self.recognition_stats['total_detections'] += len(faces)
                logger.info(f"👥 ОБНАРУЖЕНО ОБЪЕКТОВ: {len(faces)}")

                for (x, y, w, h) in faces:
                    # Расширенная проверка валидности
                    if not self.is_valid_face_region(frame, x, y, w, h):
                        filtered_count += 1
                        continue

                    face_img = frame[y:y + h, x:x + w]

                    # Проверка характеристик человека
                    if not self.validate_human_features(face_img):
                        filtered_count += 1
                        logger.debug("❌ Отфильтрован не-человеческий объект")
                        continue

                    embedding = self.get_fast_embedding(face_img)
                    if embedding is not None:
                        visitor_id, similarity = self.find_best_match(embedding)

                        processed_faces.append({
                            'coords': (x, y, w, h),
                            'embedding': embedding,
                            'similarity': similarity,
                            'visitor_id': visitor_id,
                            'status': 'detected',
                            'face_image': face_img
                        })
                    else:
                        logger.warning("❌ Не удалось получить эмбеддинг для лица")

            self.recognition_stats['filtered_detections'] += filtered_count

            return {
                'faces': processed_faces,
                'processed_count': len(processed_faces),
                'detected_count': len(faces),
                'filtered_count': filtered_count,
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"❌ Ошибка в фоновой обработке: {e}")
            return {'faces': [], 'processed_count': 0, 'detected_count': 0, 'filtered_count': 0}

    def setup_rtsp_camera(self, rtsp_url):
        """Настройка RTSP"""
        logger.info(f"📡 Подключение к камере: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

        # Пропускаем кадры для стабилизации
        for _ in range(10):
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

    def process_frame_realtime(self, frame):
        """Обработка кадра в реальном времени"""
        current_time = time.time()

        # Обновляем FPS
        self.fps_frame_count += 1
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_frame_count / (current_time - self.fps_start_time)
            self.fps_frame_count = 0
            self.fps_start_time = current_time

        # Обрабатываем с интервалом
        if current_time - self.last_processing_time < self.processing_interval:
            try:
                result, frame_time = self.results_queue.get_nowait()
                return self._apply_processing_result(frame, result, current_time)
            except:
                return frame, 0, 0, 0

        # Отправляем в фоновую обработку
        if self.frame_queue.empty():
            self.frame_queue.put((frame.copy(), current_time))

        self.last_processing_time = current_time

        # Пробуем получить результаты
        try:
            result, frame_time = self.results_queue.get_nowait()
            return self._apply_processing_result(frame, result, current_time)
        except:
            return frame, 0, 0, 0

    def _apply_processing_result(self, frame, result, current_time):
        """Применяет результаты обработки"""
        processed_frame = frame.copy()
        processed_count = 0

        # Обновляем трекинг
        tracked_faces = self.update_face_tracking(result['faces'], current_time)

        for face_data in tracked_faces:
            visitor_id = self.confirm_visitor_identity(face_data['track_id'], face_data)

            x, y, w, h = face_data['coords']
            status = face_data.get('status', 'detected')

            color = self.get_color_by_status(status)
            status_text = self.get_status_text(status)

            # Отрисовка рамки
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(processed_frame, f'{status_text}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if visitor_id:
                cv2.putText(processed_frame, f'ID: {visitor_id}', (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(processed_frame, f'Sim: {face_data["similarity"]:.2f}',
                            (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            processed_count += 1

        self.log_recognition_stats()

        return processed_frame, result['detected_count'], processed_count, result['filtered_count']

    def start_analysis(self, rtsp_url):
        """Запуск анализа"""
        logger.info("🚀 Запуск улучшенной версии с MediaPipe...")

        cap = self.setup_rtsp_camera(rtsp_url)
        if not cap.isOpened():
            return

        self.start_processing_thread()
        logger.info("✅ Анализ запущен")

        window_name = 'Trassir Analytics - ENHANCED with MediaPipe'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("📡 Потеряно соединение с камерой...")
                    time.sleep(2)
                    continue

                processed_frame, detected, processed, filtered = self.process_frame_realtime(frame)
                display_frame = self.resize_frame_for_display(processed_frame, target_width=1280)
                display_with_gallery = self.create_gallery_display(display_frame)

                # Статистика на экране
                stats_text = [
                    f"ENHANCED ANALYTICS with MediaPipe",
                    f"Objects detected: {detected}",
                    f"Faces processed: {processed}",
                    f"Filtered: {filtered}",
                    f"Active tracks: {len(self.face_tracks)}",
                    f"In gallery: {len(self.current_visitors_gallery)}",
                    f"FPS: {self.current_fps:.1f}",
                    f"Press 'q' to quit"
                ]

                overlay = display_with_gallery.copy()
                cv2.rectangle(overlay, (0, 0), (500, 220), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_with_gallery, 0.3, 0, display_with_gallery)

                for i, text in enumerate(stats_text):
                    color = (255, 255, 255)
                    if "Filtered" in text and filtered > 0:
                        color = (0, 255, 255)  # Желтый для фильтрованных объектов
                    cv2.putText(display_with_gallery, text, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.imshow(window_name, display_with_gallery)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("⏹️ Остановка по Ctrl+C...")
        finally:
            self.stop_processing = True
            if self.processing_thread:
                self.processing_thread.join(timeout=2.0)
            cap.release()
            cv2.destroyAllWindows()
            self.conn.close()

            logger.info(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
            logger.info(f"   Всего посетителей: {len(self.known_visitors_cache)}")
            logger.info(f"   Новых создано: {self.recognition_stats['new_visitors']}")
            logger.info(f"   Известных обработано: {self.recognition_stats['known_visitors']}")
            logger.info(f"   Отфильтровано объектов: {self.recognition_stats['filtered_detections']}")
            logger.info("✅ Анализ завершен")


def main():
    """Основная функция"""
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    counter = EnhancedTrassirCounter(
        processing_interval=1.0,
        similarity_threshold=0.55,
        tracking_threshold=0.45
    )

    try:
        counter.start_analysis(RTSP_URL)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()