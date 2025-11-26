# video_analytics_trassir_realistic_fixed.py
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


class RealisticTrassirCounter:
    def __init__(self, processing_interval=1.0, similarity_threshold=0.65, tracking_threshold=0.50):
        """
        Версия с реалистичными фильтрами для нормального расстояния от камеры
        """
        self.conn = sqlite3.connect('visitors_trassir_realistic.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
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
        self.photos_dir = "visitor_photos_realistic"
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

        # РЕАЛИСТИЧНЫЕ фильтры для нормального расстояния
        self.false_positive_filter = {
            'min_face_ratio': 0.008,  # ОЧЕНЬ маленькие лица (1-2% кадра)
            'max_face_ratio': 0.30,  # До 30% кадра (крупный план)
            'min_aspect_ratio': 0.5,  # Широкий диапазон пропорций
            'max_aspect_ratio': 2.0,
            'min_brightness': 15,  # Очень темные условия
            'max_brightness': 245,  # Очень яркие условия
            'edge_threshold': 10,  # Очень низкая четкость
            'required_confirmations': 2,
            'min_face_width': 40,  # Минимальная ширина в пикселях
            'min_face_height': 40  # Минимальная высота в пикселях
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
        self.frame_queue = Queue(maxsize=2)  # Ограничиваем очередь для избежания переполнения
        self.results_queue = Queue()

        # Детектор лиц с оптимизацией для дальнего расстояния
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

        logger.info("🎯 Система инициализирована с РЕАЛИСТИЧНЫМИ фильтрами для дальнего расстояния")

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
                # Логируем примерные размеры для отладки
                h, w = test_frame.shape[:2]
                logger.info(f"📐 Примерные размеры лиц на этом разрешении:")
                logger.info(f"   - Маленькое лицо (далеко): {int(w * 0.02)}x{int(h * 0.02)} пикселей")
                logger.info(f"   - Среднее лицо: {int(w * 0.08)}x{int(h * 0.08)} пикселей")
                logger.info(f"   - Крупное лицо (близко): {int(w * 0.15)}x{int(h * 0.15)} пикселей")
            else:
                logger.error("❌ Камера не передает данные")
        else:
            logger.error("❌ Не удалось подключиться к камере")

        return cap

    def analyze_face_quality(self, face_image, bbox, frame_size):
        """Анализ качества обнаруженного лица с РЕАЛИСТИЧНЫМИ критериями"""
        try:
            x, y, w, h = bbox
            frame_height, frame_width = frame_size

            # 1. Проверка АБСОЛЮТНОГО размера в пикселях (важнее чем отношение)
            logger.debug(f"📏 Абсолютный размер: {w}x{h} пикселей")

            if w < self.false_positive_filter['min_face_width']:
                self.recognition_stats['quality_rejections']['small_width'] += 1
                return False, f"Слишком маленькая ширина ({w} < {self.false_positive_filter['min_face_width']})"
            if h < self.false_positive_filter['min_face_height']:
                self.recognition_stats['quality_rejections']['small_height'] += 1
                return False, f"Слишком маленькая высота ({h} < {self.false_positive_filter['min_face_height']})"

            # 2. Проверка ОТНОСИТЕЛЬНОГО размера (вторичный критерий)
            face_area = w * h
            frame_area = frame_width * frame_height
            face_ratio = face_area / frame_area

            logger.debug(f"📐 Относительный размер: {face_ratio:.4f} ({w}x{h} пикс)")

            if face_ratio < self.false_positive_filter['min_face_ratio']:
                self.recognition_stats['quality_rejections']['small_ratio'] += 1
                # НЕ отклоняем только из-за маленького отношения, если абсолютный размер нормальный
                if w < 60 or h < 60:  # Дополнительная проверка
                    return False, f"Слишком маленькое отношение ({face_ratio:.4f})"
                else:
                    logger.debug(f"⚠️  Маленькое отношение, но нормальный размер - принимаем")

            if face_ratio > self.false_positive_filter['max_face_ratio']:
                self.recognition_stats['quality_rejections']['large_ratio'] += 1
                return False, f"Слишком большое отношение ({face_ratio:.4f})"

            # 3. Проверка соотношения сторон
            aspect_ratio = w / h
            logger.debug(f"⚖️ Соотношение сторон: {aspect_ratio:.2f}")

            if aspect_ratio < self.false_positive_filter['min_aspect_ratio']:
                self.recognition_stats['quality_rejections']['narrow'] += 1
                return False, f"Слишком узкое ({aspect_ratio:.2f})"
            if aspect_ratio > self.false_positive_filter['max_aspect_ratio']:
                self.recognition_stats['quality_rejections']['wide'] += 1
                return False, f"Слишком широкое ({aspect_ratio:.2f})"

            # 4. УСИЛЕННАЯ проверка что это действительно лицо (а не случайный объект)
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # Проверка яркости
            brightness = np.mean(gray_face)
            logger.debug(f"💡 Яркость: {brightness:.1f}")

            if brightness < self.false_positive_filter['min_brightness']:
                self.recognition_stats['quality_rejections']['dark'] += 1
                # Для маленьких лиц допускаем меньшую яркость
                if w > 80:  # Только для крупных лиц строгая проверка яркости
                    return False, f"Слишком темное ({brightness:.1f})"
                else:
                    logger.debug(f"⚠️  Темное, но маленькое лицо - принимаем")

            if brightness > self.false_positive_filter['max_brightness']:
                self.recognition_stats['quality_rejections']['bright'] += 1
                return False, f"Слишком светлое ({brightness:.1f})"

            # 5. Проверка четкости (для маленьких лиц менее строгая)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            logger.debug(f"🔍 Четкость (лапласиан): {laplacian_var:.1f}")

            # Динамический порог четкости в зависимости от размера лица
            dynamic_edge_threshold = max(self.false_positive_filter['edge_threshold'],
                                         min(30, 50 - w / 5))  # Чем меньше лицо, тем ниже требования

            if laplacian_var < dynamic_edge_threshold:
                self.recognition_stats['quality_rejections']['blurry'] += 1
                logger.debug(f"⚠️  Низкая четкость ({laplacian_var:.1f} < {dynamic_edge_threshold:.1f}), но продолжаем")
                # НЕ отклоняем из-за четкости для маленьких лиц

            # 6. Проверка контраста
            contrast = np.std(gray_face)
            logger.debug(f"🎨 Контраст: {contrast:.1f}")

            if contrast < 3:  # Очень мягкий порог
                self.recognition_stats['quality_rejections']['uniform'] += 1
                return False, f"Слишком однородная область ({contrast:.1f})"

            # 7. ДОПОЛНИТЕЛЬНАЯ проверка: гистограмма для отсеивания простых объектов
            hist = cv2.calcHist([gray_face], [0], None, [8], [0, 256])
            hist_std = np.std(hist)
            if hist_std < 100 and w > 100:  # Для крупных лиц проверяем сложность текстуры
                self.recognition_stats['quality_rejections']['simple_texture'] += 1
                return False, f"Слишком простая текстура ({hist_std:.1f})"

            return True, f"✅ Лицо {w}x{h} пикс (яркость: {brightness:.1f}, четкость: {laplacian_var:.1f})"

        except Exception as e:
            self.recognition_stats['quality_rejections']['error'] += 1
            return False, f"Ошибка анализа качества: {e}"

    def detect_faces_robust(self, frame):
        """Детекция лиц с оптимизацией для РАЗНЫХ расстояний"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_size = gray.shape

        all_faces = []

        # МНОГОУРОВНЕВАЯ детекция для разных размеров лиц
        detection_params = [
            # Уровень 1: Очень маленькие лица (далекие люди)
            {
                'scaleFactor': 1.05,
                'minNeighbors': 3,
                'minSize': (20, 20),  # Очень маленькие лица
                'maxSize': (80, 80),
                'name': 'tiny_faces'
            },
            # Уровень 2: Средние лица (нормальное расстояние)
            {
                'scaleFactor': 1.1,
                'minNeighbors': 4,
                'minSize': (40, 40),
                'maxSize': (150, 150),
                'name': 'medium_faces'
            },
            # Уровень 3: Крупные лица (близко к камере)
            {
                'scaleFactor': 1.1,
                'minNeighbors': 5,
                'minSize': (80, 80),
                'maxSize': (300, 300),
                'name': 'large_faces'
            }
        ]

        total_raw_detections = 0
        valid_count = 0
        rejected_count = 0

        for params in detection_params:
            # Детекция основным каскадом
            faces1 = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=params['scaleFactor'],
                minNeighbors=params['minNeighbors'],
                minSize=params['minSize'],
                maxSize=params['maxSize'],
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Детекция альтернативным каскадом
            faces2 = self.alt_face_cascade.detectMultiScale(
                gray,
                scaleFactor=params['scaleFactor'],
                minNeighbors=params['minNeighbors'] - 1,  # Более чувствительный
                minSize=params['minSize'],
                maxSize=params['maxSize'],
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            total_raw_detections += len(faces1) + len(faces2)
            logger.debug(f"🔍 {params['name']}: основной {len(faces1)}, альтернативный {len(faces2)}")

            # Объединяем и фильтруем результаты
            face_set = set()

            for faces in [faces1, faces2]:
                for (x, y, w, h) in faces:
                    # Группировка близких детекций
                    face_key = (x // 15, y // 15, w // 15, h // 15)
                    if face_key in face_set:
                        continue

                    face_set.add(face_key)

                    # Проверка качества лица
                    face_roi = frame[y:y + h, x:x + w]
                    is_valid, quality_msg = self.analyze_face_quality(face_roi, (x, y, w, h), frame_size)

                    if is_valid:
                        all_faces.append((x, y, w, h))
                        valid_count += 1
                        logger.info(f"✅ Принято {params['name']} {w}x{h}: {quality_msg}")
                    else:
                        rejected_count += 1
                        if w >= 40:  # Логируем только значительные отклонения
                            logger.info(f"❌ Отклонено {params['name']} {w}x{h}: {quality_msg}")

        logger.info(
            f"📊 МНОГОУРОВНЕВАЯ детекция: сырых {total_raw_detections}, принято {valid_count}, отклонено {rejected_count}")

        with self.stats_lock:
            self.recognition_stats['rejected_detections'] += rejected_count

        return all_faces

    def get_fast_embedding(self, face_image):
        """Получение эмбеддинга с адаптацией под маленькие лица"""
        try:
            # Адаптивный ресайз в зависимости от размера лица
            h, w = face_image.shape[:2]

            if w < 60 or h < 60:
                # Для маленьких лиц используем меньший размер для сохранения деталей
                target_size = max(80, min(w, h))
                face_resized = cv2.resize(face_image, (target_size, target_size))
                logger.debug(f"🔍 Маленькое лицо {w}x{h}, ресайз до {target_size}x{target_size}")
            else:
                # Для нормальных лиц стандартный размер
                face_resized = cv2.resize(face_image, (160, 160))

            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Упрощенная обработка для скорости
            result = DeepFace.represent(
                face_rgb,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='opencv',
                align=False
            )

            embedding = np.array(result[0]['embedding'], dtype=np.float32)

            # Мягкая проверка качества
            if np.all(embedding == 0):
                logger.warning("❌ Нулевой эмбеддинг")
                return None

            norm = np.linalg.norm(embedding)
            if norm < 0.005:  # Очень мягкий порог
                logger.warning(f"❌ Слишком маленькая норма эмбеддинга: {norm}")
                return None

            logger.debug(f"✅ Эмбеддинг получен, норма: {norm:.4f}")
            return embedding

        except Exception as e:
            logger.warning(f"❌ Ошибка получения эмбеддинга: {e}")
            return None

    def calculate_similarity(self, embedding1, embedding2):
        """Вычисление косинусного сходства между эмбеддингами"""
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0

            # Нормализация векторов
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2

            # Косинусное сходство
            similarity = np.dot(embedding1_norm, embedding2_norm)

            return float(similarity)
        except Exception as e:
            logger.warning(f"❌ Ошибка вычисления сходства: {e}")
            return 0.0

    def find_best_match(self, embedding):
        """Поиск лучшего совпадения в базе данных"""
        best_match_id = None
        best_similarity = 0.0

        for visitor_id, known_embedding in self.known_visitors_cache.items():
            similarity = self.calculate_similarity(embedding, known_embedding)

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match_id = visitor_id

        return best_match_id, best_similarity

    def update_face_tracking(self, faces, current_time):
        """Обновление системы трекинга лиц"""
        active_tracks = {}

        with self.tracks_lock:
            # Очистка старых треков
            for track_id, track_info in list(self.face_tracks.items()):
                if current_time - track_info['last_seen'] > self.track_max_age:
                    logger.info(f"🗑️ Удален старый трек {track_id}")
                    del self.face_tracks[track_id]

            # Обновление существующих треков
            for face_bbox in faces:
                x, y, w, h = face_bbox
                face_center = (x + w // 2, y + h // 2)

                best_track_id = None
                best_distance = float('inf')

                # Поиск ближайшего существующего трека
                for track_id, track_info in self.face_tracks.items():
                    if current_time - track_info['last_seen'] > 1.0:
                        continue

                    last_center = track_info['last_center']
                    distance = math.sqrt((face_center[0] - last_center[0]) ** 2 +
                                         (face_center[1] - last_center[1]) ** 2)

                    # Пороговое расстояние для ассоциации
                    max_distance = min(w, h) * 1.5

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
                    logger.info(f"🆕 Создан новый трек {track_id}")

        return active_tracks

    def save_visitor_photo(self, face_image, visitor_id):
        """Сохранение фото посетителя"""
        try:
            filename = f"visitor_{visitor_id}_{int(time.time())}.jpg"
            filepath = os.path.join(self.photos_dir, self.current_session_dir, filename)

            cv2.imwrite(filepath, face_image)
            logger.info(f"📸 Сохранено фото: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения фото: {e}")
            return None

    def update_visitor_database(self, embedding, face_image, track_id):
        """Обновление базы данных посетителей"""
        try:
            # Поиск совпадения
            visitor_id, similarity = self.find_best_match(embedding)
            current_time = datetime.datetime.now()

            if visitor_id is not None:
                # Известный посетитель
                cursor = self.conn.cursor()
                cursor.execute('''
                    UPDATE visitors 
                    SET last_seen = ?, visit_count = visit_count + 1, last_updated = ?
                    WHERE id = ?
                ''', (current_time, current_time, visitor_id))
                self.conn.commit()

                with self.stats_lock:
                    self.recognition_stats['known_visitors'] += 1

                logger.info(f"👤 Обновлен известный посетитель {visitor_id} (сходство: {similarity:.3f})")
                return visitor_id, 'known'
            else:
                # Новый посетитель
                cursor = self.conn.cursor()
                photo_path = self.save_visitor_photo(face_image, self.next_track_id)

                cursor.execute('''
                    INSERT INTO visitors 
                    (face_embedding, first_seen, last_seen, last_updated, photo_path)
                    VALUES (?, ?, ?, ?, ?)
                ''', (embedding.tobytes(), current_time, current_time, current_time, photo_path))

                new_visitor_id = cursor.lastrowid
                self.conn.commit()

                # Обновление кэша
                self.known_visitors_cache[new_visitor_id] = embedding

                with self.stats_lock:
                    self.recognition_stats['new_visitors'] += 1

                logger.info(f"🆕 Добавлен новый посетитель {new_visitor_id}")
                return new_visitor_id, 'new'

        except Exception as e:
            logger.error(f"❌ Ошибка обновления БД: {e}")
            return None, 'error'

    def process_frame_realtime(self, frame):
        """Обработка кадра в реальном времени"""
        current_time = time.time()

        # Обновление FPS
        self.fps_frame_count += 1
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_frame_count / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
            self.fps_frame_count = 0

        # Детекция лиц
        faces = self.detect_faces_robust(frame)

        # Обновление трекинга
        active_tracks = self.update_face_tracking(faces, current_time)

        # Отрисовка результатов
        processed_frame = frame.copy()

        detected_count = 0
        processed_count = 0

        for track_id, track_info in active_tracks.items():
            x, y, w, h = track_info['bbox']

            # Определение статуса для цвета
            if track_info.get('visitor_id'):
                status = 'known' if track_info.get('status') == 'known' else 'new'
            elif track_info['confirmed_count'] >= self.false_positive_filter['required_confirmations']:
                status = 'tracking'
            else:
                status = 'detected'

            color = self.COLORS[status]

            # Отрисовка bounding box
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)

            # Текст с информацией
            label = f"ID:{track_id} {status}"
            if track_info.get('visitor_id'):
                label += f" V:{track_info['visitor_id']}"

            cv2.putText(processed_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detected_count += 1

            # Обработка лица для распознавания (только для подтвержденных треков)
            if (track_info['confirmed_count'] >= self.false_positive_filter['required_confirmations'] and
                    current_time - track_info.get('last_processed', 0) > self.processing_interval):

                try:
                    face_roi = frame[y:y + h, x:x + w]
                    embedding = self.get_fast_embedding(face_roi)

                    if embedding is not None:
                        visitor_id, status = self.update_visitor_database(embedding, face_roi, track_id)
                        if visitor_id:
                            track_info['visitor_id'] = visitor_id
                            track_info['status'] = status
                            processed_count += 1

                    track_info['last_processed'] = current_time

                except Exception as e:
                    logger.error(f"❌ Ошибка обработки лица: {e}")

        with self.stats_lock:
            self.recognition_stats['total_detections'] += len(faces)
            self.recognition_stats['valid_detections'] += detected_count
            self.recognition_stats['frames_processed'] += 1

        return processed_frame, detected_count, processed_count

    def resize_frame_for_display(self, frame, target_width=1280):
        """Изменение размера кадра для отображения"""
        try:
            h, w = frame.shape[:2]
            if w <= target_width:
                return frame

            scale_factor = target_width / w
            new_width = target_width
            new_height = int(h * scale_factor)

            return cv2.resize(frame, (new_width, new_height))
        except Exception as e:
            logger.error(f"❌ Ошибка изменения размера: {e}")
            return frame

    def create_gallery_display(self, main_frame):
        """Создание отображения с галереей посетителей"""
        try:
            main_height, main_width = main_frame.shape[:2]

            # Создание панели для галереи
            gallery_height = 180
            combined_width = main_width

            # Создание комбинированного изображения
            combined_frame = np.zeros((main_height + gallery_height, combined_width, 3), dtype=np.uint8)
            combined_frame[0:main_height, 0:main_width] = main_frame

            # Заголовок галереи
            cv2.putText(combined_frame, "Current Visitors Gallery", (10, main_height + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Очистка старой галереи
            current_time = time.time()
            if current_time - self.last_gallery_cleanup > self.gallery_cleanup_interval:
                with self.gallery_lock:
                    self.current_visitors_gallery = {
                        k: v for k, v in self.current_visitors_gallery.items()
                        if current_time - v['last_seen'] < 300  # 5 минут
                    }
                self.last_gallery_cleanup = current_time

            # Отрисовка галереи
            gallery_x = 10
            for track_id, visitor_info in list(self.current_visitors_gallery.items())[:8]:
                if 'photo' in visitor_info:
                    photo = visitor_info['photo']
                    photo_resized = cv2.resize(photo, (120, 120))

                    # Размещение фото в галерее
                    y_start = main_height + 40
                    y_end = y_start + 120
                    x_end = gallery_x + 120

                    if x_end < combined_width:
                        combined_frame[y_start:y_end, gallery_x:gallery_x + 120] = photo_resized

                        # Подпись
                        label = f"ID:{visitor_info.get('visitor_id', track_id)}"
                        cv2.putText(combined_frame, label, (gallery_x, main_height + 165),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        gallery_x += 130

            return combined_frame

        except Exception as e:
            logger.error(f"❌ Ошибка создания галереи: {e}")
            return main_frame

    def processing_worker(self):
        """Рабочий поток для обработки кадров"""
        logger.info("🔄 Запущен поток обработки")

        while not self.stop_processing:
            try:
                # Получение кадра из очереди с таймаутом
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:
                    break

                frame, frame_time = frame_data

                # Обработка кадра
                processed_frame, detected, processed = self.process_frame_realtime(frame)

                # Помещение результата в очередь
                if not self.results_queue.full():
                    self.results_queue.put((processed_frame, detected, processed))

                self.frame_queue.task_done()

            except Exception as e:
                logger.error(f"❌ Ошибка в потоке обработки: {e}")
                time.sleep(0.1)

        logger.info("🛑 Поток обработки остановлен")

    def start_processing_thread(self):
        """Запуск потока обработки"""
        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("✅ Поток обработки запущен")

    def start_analysis(self, rtsp_url):
        """Запуск анализа с РЕАЛИСТИЧНЫМИ настройками"""
        logger.info("🚀 Запуск версии с РЕАЛИСТИЧНЫМИ фильтрами для любого расстояния...")

        cap = self.setup_rtsp_camera(rtsp_url)
        if not cap.isOpened():
            return

        self.start_processing_thread()
        logger.info("✅ Анализ запущен - система принимает лица любого размера!")
        logger.info("💡 Система настроена на обнаружение лиц от 20x20 до 300x300 пикселей")

        window_name = 'Trassir Analytics - ANY DISTANCE'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("📡 Потеряно соединение с камерой...")
                    time.sleep(2)
                    continue

                # Обработка кадра в основном потоке (упрощенная версия)
                processed_frame, detected, processed = self.process_frame_realtime(frame)
                display_frame = self.resize_frame_for_display(processed_frame, target_width=1280)
                display_with_gallery = self.create_gallery_display(display_frame)

                # Статистика на экране
                stats_text = [
                    f"ANY DISTANCE - REALISTIC FILTERS",
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
            logger.info(f"   Отклонено: {self.recognition_stats['rejected_detections']}")
            if self.recognition_stats['quality_rejections']:
                logger.info(f"   Детальная статистика отклонений: {dict(self.recognition_stats['quality_rejections'])}")
            logger.info("✅ Анализ завершен")


def main():
    """Основная функция"""
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    counter = RealisticTrassirCounter(
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