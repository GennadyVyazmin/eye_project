# video_analytics_trassir_adjusted.py
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedTrassirCounter:
    def __init__(self, processing_interval=1.0, similarity_threshold=0.65, tracking_threshold=0.50):
        """
        Версия с ослабленными фильтрами для тестирования
        """
        self.conn = sqlite3.connect('visitors_trassir_improved.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
        self.similarity_threshold = similarity_threshold
        self.tracking_threshold = tracking_threshold

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
        self.photos_dir = "visitor_photos_improved"
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
        self.gallery_max_size = 8
        self.gallery_cleanup_interval = 60.0
        self.last_gallery_cleanup = time.time()
        self.photo_size = (120, 160)

        # ОСЛАБЛЕННЫЕ фильтры для тестирования
        self.false_positive_filter = {
            'min_face_ratio': 0.02,  # СНИЖЕНО: было 0.08
            'max_face_ratio': 0.60,  # ПОВЫШЕНО: было 0.40
            'min_aspect_ratio': 0.5,  # СНИЖЕНО: было 0.7
            'max_aspect_ratio': 2.0,  # ПОВЫШЕНО: было 1.4
            'min_brightness': 20,  # СНИЖЕНО: было 30
            'max_brightness': 240,  # ПОВЫШЕНО: было 220
            'edge_threshold': 20,  # СНИЖЕНО: было 50
            'required_confirmations': 2  # СНИЖЕНО: было 3
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

        # Очередь для обработки
        self.frame_queue = Queue(maxsize=1)
        self.results_queue = Queue()

        # Детектор лиц
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.alt_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )

        # Предзагрузка известных посетителей
        self._load_known_visitors()

        # Поток для обработки
        self.processing_thread = None
        self.stop_processing = False

        # Для расчета FPS
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

        logger.info("🎯 Система инициализирована с ОСЛАБЛЕННЫМИ фильтрами")

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
        """Настройка RTSP подключения"""
        logger.info(f"📡 Подключение к камере: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

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

    def analyze_face_quality(self, face_image, bbox, frame_size):
        """Анализ качества обнаруженного лица с отладкой"""
        try:
            x, y, w, h = bbox
            frame_height, frame_width = frame_size

            # 1. Проверка размера лица относительно кадра
            face_area = w * h
            frame_area = frame_width * frame_height
            face_ratio = face_area / frame_area

            logger.debug(f"📏 Размер лица: {w}x{h}, отношение: {face_ratio:.4f}")

            if face_ratio < self.false_positive_filter['min_face_ratio']:
                self.recognition_stats['quality_rejections']['small_size'] += 1
                return False, f"Слишком маленькое лицо ({face_ratio:.4f} < {self.false_positive_filter['min_face_ratio']})"
            if face_ratio > self.false_positive_filter['max_face_ratio']:
                self.recognition_stats['quality_rejections']['large_size'] += 1
                return False, f"Слишком большое лицо ({face_ratio:.4f} > {self.false_positive_filter['max_face_ratio']})"

            # 2. Проверка соотношения сторон
            aspect_ratio = w / h
            logger.debug(f"⚖️ Соотношение сторон: {aspect_ratio:.2f}")

            if aspect_ratio < self.false_positive_filter['min_aspect_ratio']:
                self.recognition_stats['quality_rejections']['narrow'] += 1
                return False, f"Слишком узкое ({aspect_ratio:.2f} < {self.false_positive_filter['min_aspect_ratio']})"
            if aspect_ratio > self.false_positive_filter['max_aspect_ratio']:
                self.recognition_stats['quality_rejections']['wide'] += 1
                return False, f"Слишком широкое ({aspect_ratio:.2f} > {self.false_positive_filter['max_aspect_ratio']})"

            # 3. Проверка яркости
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_face)
            logger.debug(f"💡 Яркость: {brightness:.1f}")

            if brightness < self.false_positive_filter['min_brightness']:
                self.recognition_stats['quality_rejections']['dark'] += 1
                return False, f"Слишком темное ({brightness:.1f} < {self.false_positive_filter['min_brightness']})"
            if brightness > self.false_positive_filter['max_brightness']:
                self.recognition_stats['quality_rejections']['bright'] += 1
                return False, f"Слишком светлое ({brightness:.1f} > {self.false_positive_filter['max_brightness']})"

            # 4. Проверка четкости (лапласиан)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            logger.debug(f"🔍 Четкость (лапласиан): {laplacian_var:.1f}")

            if laplacian_var < self.false_positive_filter['edge_threshold']:
                self.recognition_stats['quality_rejections']['blurry'] += 1
                return False, f"Нечеткое ({laplacian_var:.1f} < {self.false_positive_filter['edge_threshold']})"

            # 5. Проверка заполненности области
            contrast = np.std(gray_face)
            logger.debug(f"🎨 Контраст: {contrast:.1f}")

            if contrast < 5:  # Еще более мягкий порог
                self.recognition_stats['quality_rejections']['uniform'] += 1
                return False, f"Слишком однородная область ({contrast:.1f})"

            return True, f"✅ Качество OK (яркость: {brightness:.1f}, четкость: {laplacian_var:.1f}, контраст: {contrast:.1f})"

        except Exception as e:
            self.recognition_stats['quality_rejections']['error'] += 1
            return False, f"Ошибка анализа качества: {e}"

    def detect_faces_robust(self, frame):
        """Надежная детекция лиц с ОСЛАБЛЕННОЙ фильтрацией"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_size = gray.shape

        all_faces = []

        # ПЕРВЫЙ ПРОХОД: Детекция без фильтрации для отладки
        faces1 = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # Более агрессивный поиск
            minNeighbors=4,  # Меньше соседей для большей чувствительности
            minSize=(30, 30),  # Меньший минимальный размер
            maxSize=(400, 400),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        faces2 = self.alt_face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,  # Еще меньше соседей
            minSize=(25, 25),  # Еще меньший размер
            maxSize=(500, 500),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        logger.info(f"🔍 СЫРАЯ детекция: основной {len(faces1)}, альтернативный {len(faces2)}")

        # Объединяем и фильтруем результаты
        face_set = set()
        valid_count = 0
        rejected_count = 0

        for faces in [faces1, faces2]:
            for (x, y, w, h) in faces:
                # Группировка близких детекций
                face_key = (x // 10, y // 10, w // 10, h // 10)  # Более точная группировка
                if face_key in face_set:
                    continue

                face_set.add(face_key)

                # Проверка качества лица
                face_roi = frame[y:y + h, x:x + w]
                is_valid, quality_msg = self.analyze_face_quality(face_roi, (x, y, w, h), frame_size)

                if is_valid:
                    all_faces.append((x, y, w, h))
                    valid_count += 1
                    logger.info(f"✅ Принято лицо {w}x{h}: {quality_msg}")
                else:
                    rejected_count += 1
                    logger.info(f"❌ Отклонено {w}x{h}: {quality_msg}")

        logger.info(f"📊 ИТОГО: принято {valid_count}, отклонено {rejected_count}")
        self.recognition_stats['rejected_detections'] += rejected_count

        return all_faces

    def get_fast_embedding(self, face_image):
        """Получение эмбеддинга с УПРОЩЕННОЙ проверкой"""
        try:
            # Упрощенная предобработка
            face_resized = cv2.resize(face_image, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Минимальная обработка для скорости
            result = DeepFace.represent(
                face_rgb,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='opencv',
                align=False  # Отключаем выравнивание для скорости
            )

            embedding = np.array(result[0]['embedding'], dtype=np.float32)

            # ОЧЕНЬ мягкая проверка качества
            if np.all(embedding == 0):
                logger.warning("❌ Нулевой эмбеддинг")
                return None

            norm = np.linalg.norm(embedding)
            if norm < 0.01:  # Очень мягкий порог
                logger.warning(f"❌ Слишком маленькая норма эмбеддинга: {norm}")
                return None

            logger.debug(f"✅ Эмбеддинг получен, норма: {norm:.4f}")
            return embedding

        except Exception as e:
            logger.warning(f"❌ Ошибка получения эмбеддинга: {e}")
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
            logger.debug(f"📐 Схожесть: {similarity:.3f}")
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

        logger.debug(f"🔍 Лучшее совпадение: ID {best_match_id}, схожесть {best_similarity:.3f}")
        return best_match_id, best_similarity

    def cleanup_old_gallery_entries(self):
        """Очистка старых записей из галереи"""
        current_time = time.time()
        if current_time - self.last_gallery_cleanup >= self.gallery_cleanup_interval:
            removed_count = 0
            for visitor_id in list(self.current_visitors_gallery.keys()):
                last_seen = self.current_visitors_gallery[visitor_id]['last_seen']
                if current_time - last_seen > self.gallery_cleanup_interval:
                    del self.current_visitors_gallery[visitor_id]
                    removed_count += 1

            if removed_count > 0:
                logger.info(f"🧹 Очистка галереи: удалено {removed_count} старых записей")

            self.last_gallery_cleanup = current_time

    def update_face_tracking(self, current_faces, timestamp):
        """Обновление трекинга лиц между кадрами"""
        updated_faces = []

        # Очищаем старые треки
        for track_id in list(self.face_tracks.keys()):
            if timestamp - self.face_tracks[track_id]['last_seen'] > self.track_max_age:
                logger.debug(f"🗑️ Удален старый трек {track_id}")
                del self.face_tracks[track_id]

        for face_data in current_faces:
            embedding = face_data['embedding']
            coords = face_data['coords']
            face_image = face_data['face_image']

            best_track_id = None
            best_similarity = 0.0

            # Ищем совпадение с существующими треками
            for track_id, track_data in self.face_tracks.items():
                similarity = self.calculate_similarity(embedding, track_data['embedding'])
                if similarity > best_similarity and similarity > self.tracking_threshold:
                    best_similarity = similarity
                    best_track_id = track_id

            if best_track_id is not None:
                # Обновляем существующий трек
                track_data = self.face_tracks[best_track_id]
                track_data.update({
                    'embedding': embedding,
                    'last_seen': timestamp,
                    'coords': coords,
                    'face_image': face_image,
                    'confirmation_count': track_data.get('confirmation_count', 0) + 1
                })
                face_data['track_id'] = best_track_id
                face_data['visitor_id'] = track_data.get('visitor_id')
                face_data['status'] = 'tracking'
                logger.debug(f"🔄 Обновлен трек {best_track_id}, подтверждений: {track_data['confirmation_count']}")
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
                    'created_at': timestamp,
                    'confirmation_count': 1
                }
                face_data['track_id'] = track_id
                face_data['status'] = 'tracking'
                logger.info(f"🎯 Создан новый трек {track_id}")

            updated_faces.append(face_data)

        return updated_faces

    def confirm_visitor_identity(self, track_id, face_data):
        """Подтверждение идентичности посетителя с УПРОЩЕННОЙ логикой"""
        track_data = self.face_tracks.get(track_id)
        if not track_data:
            return None

        embedding = face_data['embedding']
        face_image = face_data['face_image']
        timestamp = time.time()

        # Проверяем количество подтверждений
        confirmation_count = track_data.get('confirmation_count', 0)
        min_confirmations = self.false_positive_filter['required_confirmations']

        if confirmation_count < min_confirmations:
            face_data['status'] = 'analyzing'
            logger.debug(f"⏳ Трек {track_id} ожидает подтверждений: {confirmation_count}/{min_confirmations}")
            return None

        # Если уже есть visitor_id, обновляем его
        if track_data['visitor_id']:
            visitor_id = track_data['visitor_id']
            self.recognition_stats['known_visitors'] += 1
            face_data['status'] = 'known'
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
            self.update_visitor_gallery(visitor_id, face_image, 'known')
            logger.info(f"👤 ОПОЗНАН известный посетитель {visitor_id}, схожесть: {similarity:.3f}")
            return visitor_id
        else:
            # Создаем нового посетителя
            track_duration = timestamp - track_data['created_at']
            if track_duration > 2.0:  # Уменьшено время ожидания
                new_visitor_id = self._create_new_visitor(embedding, face_image, track_id)
                if new_visitor_id:
                    self.recognition_stats['new_visitors'] += 1
                    face_data['status'] = 'new'
                    self.update_visitor_gallery(new_visitor_id, face_image, 'new')
                    logger.info(f"🆕 СОЗДАН новый посетитель {new_visitor_id}, схожесть с известными: {similarity:.3f}")
                return new_visitor_id

        return None

    def _create_new_visitor(self, embedding, face_image, track_id):
        """Создание нового посетителя"""
        cursor = self.conn.cursor()
        now = datetime.datetime.now()

        visitor_id = None
        try:
            embedding_blob = embedding.astype(np.float32).tobytes()

            # Создаем временное фото для получения ID
            photo_path = self.save_visitor_photo(face_image, "temp")

            cursor.execute(
                """INSERT INTO visitors (face_embedding, first_seen, last_seen, 
                   visit_count, last_updated, confirmed_count, photo_path, quality_score) 
                   VALUES (?, ?, ?, 1, ?, 1, ?, 1.0)""",
                (embedding_blob, now, now, now, photo_path)
            )
            visitor_id = cursor.lastrowid

            # Обновляем фото с правильным ID
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
            if width < 100:  # Уменьшен минимальный размер
                scale = 100 / width
                new_width = 100
                new_height = int(height * scale)
                photo_clean = cv2.resize(photo_clean, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            filename = f"visitor_{visitor_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(self.photos_dir, filename)
            session_filepath = os.path.join(self.photos_dir, self.current_session_dir, filename)

            cv2.imwrite(filepath, photo_clean)
            cv2.imwrite(session_filepath, photo_clean)

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
            'rejected': 'REJECTED'
        }
        return status_texts.get(status, 'UNKNOWN')

    def get_color_by_status(self, status):
        """Получение цвета по статусу"""
        return self.COLORS.get(status, (255, 255, 255))

    def create_gallery_display(self, main_frame):
        """Создание галереи посетителей с автоочисткой"""
        try:
            # Очищаем старые записи
            self.cleanup_old_gallery_entries()

            main_height, main_width = main_frame.shape[:2]
            gallery_width = 300
            gallery_panel = np.zeros((main_height, gallery_width, 3), dtype=np.uint8)

            cv2.putText(gallery_panel, "CURRENT VISITORS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(gallery_panel, f"Active: {len(self.current_visitors_gallery)}", (10, 60),
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
                cv2.putText(gallery_panel, "No active", (50, main_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
                cv2.putText(gallery_panel, "visitors", (60, main_height // 2 + 30),
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
        if current_time - self.last_log_time >= 5.0:
            logger.info(f"📊 СТАТИСТИКА: Всего в базе: {len(self.known_visitors_cache)}, "
                        f"Активных треков: {len(self.face_tracks)}, "
                        f"В галерее: {len(self.current_visitors_gallery)}, "
                        f"Новых за сессию: {self.recognition_stats['new_visitors']}, "
                        f"Известных: {self.recognition_stats['known_visitors']}, "
                        f"Отклонено: {self.recognition_stats['rejected_detections']}")

            # Детальная статистика по причинам отклонения
            if self.recognition_stats['quality_rejections']:
                logger.info(f"📋 ПРИЧИНЫ ОТКЛОНЕНИЯ: {dict(self.recognition_stats['quality_rejections'])}")

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
        """Тяжелые операции обработки с ОСЛАБЛЕННОЙ фильтрацией"""
        try:
            # Детекция лиц с фильтрацией
            faces = self.detect_faces_robust(frame)

            processed_faces = []
            if len(faces) > 0:
                self.recognition_stats['total_detections'] += len(faces)
                self.recognition_stats['valid_detections'] += len(faces)
                logger.info(f"👥 ОБНАРУЖЕНО ЛИЦ: {len(faces)}")

                for (x, y, w, h) in faces:
                    face_img = frame[y:y + h, x:x + w]

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
                        logger.debug("❌ Пропущено лицо из-за низкого качества эмбеддинга")

            return {
                'faces': processed_faces,
                'processed_count': len(processed_faces),
                'detected_count': len(faces),
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"❌ Ошибка в фоновой обработке: {e}")
            return {'faces': [], 'processed_count': 0, 'detected_count': 0}

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
                return frame, 0, 0

        # Отправляем в фоновую обработку
        if self.frame_queue.empty():
            self.frame_queue.put((frame.copy(), current_time))

        self.last_processing_time = current_time

        # Пробуем получить результаты
        try:
            result, frame_time = self.results_queue.get_nowait()
            return self._apply_processing_result(frame, result, current_time)
        except:
            return frame, 0, 0

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

        return processed_frame, result['detected_count'], processed_count

    def start_analysis(self, rtsp_url):
        """Запуск анализа"""
        logger.info("🚀 Запуск версии с ОСЛАБЛЕННЫМИ фильтрами...")

        cap = self.setup_rtsp_camera(rtsp_url)
        if not cap.isOpened():
            return

        self.start_processing_thread()
        logger.info("✅ Анализ запущен с ОСЛАБЛЕННЫМИ фильтрами")

        window_name = 'Trassir Analytics - RELAXED FILTERS'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("📡 Потеряно соединение с камерой...")
                    time.sleep(2)
                    continue

                processed_frame, detected, processed = self.process_frame_realtime(frame)
                display_frame = self.resize_frame_for_display(processed_frame, target_width=1280)
                display_with_gallery = self.create_gallery_display(display_frame)

                # Статистика на экране
                stats_text = [
                    f"RELAXED FILTERS - TEST MODE",
                    f"Valid detections: {detected}",
                    f"Visitors processed: {processed}",
                    f"Active tracks: {len(self.face_tracks)}",
                    f"In gallery: {len(self.current_visitors_gallery)}",
                    f"Rejected: {self.recognition_stats['rejected_detections']}",
                    f"FPS: {self.current_fps:.1f}",
                    f"Press 'q' to quit"
                ]

                overlay = display_with_gallery.copy()
                cv2.rectangle(overlay, (0, 0), (550, 220), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_with_gallery, 0.3, 0, display_with_gallery)

                for i, text in enumerate(stats_text):
                    cv2.putText(display_with_gallery, text, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
            logger.info(f"   Отклонено ложных срабатываний: {self.recognition_stats['rejected_detections']}")
            if self.recognition_stats['quality_rejections']:
                logger.info(f"   Детальная статистика отклонений: {dict(self.recognition_stats['quality_rejections'])}")
            logger.info("✅ Анализ завершен")


def main():
    """Основная функция"""
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    counter = ImprovedTrassirCounter(
        processing_interval=1.0,
        similarity_threshold=0.65,
        tracking_threshold=0.50
    )

    try:
        counter.start_analysis(RTSP_URL)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()