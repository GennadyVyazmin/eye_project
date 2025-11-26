# video_analytics_trassir_lightweight.py
import cv2
import numpy as np
import sqlite3
import datetime
import time
import logging
import threading
from queue import Queue
import os
import mediapipe as mp

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


class LightweightTrassirCounter:
    def __init__(self, processing_interval=1.0, tracking_threshold=0.6):
        """
        Облегченная версия без DeepFace, только MediaPipe + OpenCV
        """
        self.conn = sqlite3.connect('visitors_trassir_lightweight.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
        self.tracking_threshold = tracking_threshold

        # Параметры фильтрации
        self.min_face_size = 80
        self.max_face_size = 400
        self.min_confidence = 0.6

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
        self.photos_dir = "visitor_photos_lightweight"
        self.current_session_dir = "current_session"
        self._create_directories()

        # Трекинг состояния
        self.last_processing_time = 0
        self.known_visitors = {}  # Простая структура для хранения известных лиц
        self.next_visitor_id = 1

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

        # Инициализация MediaPipe
        self.setup_mediapipe()

        # Для расчета FPS
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

        logger.info("🎯 Облегченная система инициализирована (только MediaPipe)")

    def setup_mediapipe(self):
        """Инициализация MediaPipe"""
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 для ближних, 1 для дальних лиц
                min_detection_confidence=0.5
            )
            logger.info("✅ MediaPipe инициализирован для детекции лиц")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации MediaPipe: {e}")
            raise

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
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                visit_count INTEGER DEFAULT 1,
                last_updated TIMESTAMP,
                photo_path TEXT,
                track_features BLOB
            )
        ''')
        self.conn.commit()

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

                confidence = detection.score[0]

                # Фильтрация по уверенности и размеру
                if (confidence >= self.min_confidence and
                        width >= self.min_face_size and height >= self.min_face_size and
                        width <= self.max_face_size and height <= self.max_face_size):

                    # Дополнительная проверка на валидность
                    if self.is_valid_face_region(frame, x, y, width, height):
                        faces.append({
                            'coords': (x, y, width, height),
                            'confidence': confidence,
                            'keypoints': detection.location_data.relative_keypoints
                        })

        return faces

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

    def extract_simple_features(self, face_image):
        """Извлечение простых признаков для трекинга без DeepFace"""
        try:
            # Ресайз для единообразия
            resized = cv2.resize(face_image, (64, 64))

            # Конвертация в grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Гистограмма градиентов (упрощенная версия)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

            mag, ang = cv2.cartToPolar(gx, gy)

            # Упрощенные признаки: средняя яркость, контраст, гистограмма
            features = []
            features.append(np.mean(gray))  # Средняя яркость
            features.append(np.std(gray))  # Контраст
            features.extend(np.histogram(gray, bins=8)[0])  # Гистограмма

            # Нормализация
            features = np.array(features, dtype=np.float32)
            if np.linalg.norm(features) > 0:
                features = features / np.linalg.norm(features)

            return features
        except Exception as e:
            logger.warning(f"Ошибка извлечения признаков: {e}")
            return None

    def calculate_feature_similarity(self, features1, features2):
        """Расчет схожести на основе простых признаков"""
        if features1 is None or features2 is None:
            return 0.0

        try:
            # Косинусная схожесть
            similarity = np.dot(features1, features2)
            return max(0.0, min(1.0, similarity))
        except:
            return 0.0

    def update_face_tracking(self, current_faces, timestamp):
        """Обновление трекинга лиц между кадрами"""
        updated_faces = []

        for face_data in current_faces:
            features = face_data['features']
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
                similarity = self.calculate_feature_similarity(features, track_data['features'])
                if similarity > best_similarity and similarity > self.tracking_threshold:
                    best_similarity = similarity
                    best_track_id = track_id

            if best_track_id is not None:
                # Обновляем существующий трек
                self.face_tracks[best_track_id].update({
                    'features': features,
                    'last_seen': timestamp,
                    'coords': coords,
                    'face_image': face_image
                })
                face_data['track_id'] = best_track_id
                face_data['visitor_id'] = self.face_tracks[best_track_id].get('visitor_id')
                face_data['status'] = 'tracking'
                face_data['similarity'] = best_similarity
                logger.debug(f"🔄 Обновлен трек {best_track_id}, схожесть: {best_similarity:.3f}")
            else:
                # Создаем новый трек
                track_id = self.next_track_id
                self.next_track_id += 1
                self.face_tracks[track_id] = {
                    'features': features,
                    'last_seen': timestamp,
                    'coords': coords,
                    'face_image': face_image,
                    'visitor_id': None,
                    'created_at': timestamp
                }
                face_data['track_id'] = track_id
                face_data['status'] = 'tracking'
                face_data['similarity'] = 0.0
                logger.info(f"🎯 Создан новый трек {track_id}")

            updated_faces.append(face_data)

        return updated_faces

    def confirm_visitor_identity(self, track_id, face_data):
        """Подтверждение идентичности посетителя"""
        track_data = self.face_tracks.get(track_id)
        if not track_data:
            return None

        features = face_data['features']
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
        visitor_id, similarity = self.find_best_match(features)

        if similarity > 0.7:  # Более высокий порог для простых признаков
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
            if track_duration > 3.0:  # Увеличили время для более надежного трекинга
                new_visitor_id = self._create_new_visitor(features, face_image, track_id)
                if new_visitor_id:
                    self.recognition_stats['new_visitors'] += 1
                    face_data['status'] = 'new'

                    # Сохраняем фото и обновляем галерею
                    self.update_visitor_gallery(new_visitor_id, face_image, 'new')

                    logger.info(f"🆕 СОЗДАН новый посетитель {new_visitor_id}")
                return new_visitor_id
            else:
                face_data['status'] = 'analyzing'
                logger.debug(f"⏳ Трек {track_id} ожидает подтверждения ({track_duration:.1f}s)")

        return None

    def find_best_match(self, features):
        """Поиск лучшего совпадения среди известных посетителей"""
        if features is None:
            return None, 0.0

        best_match_id = None
        best_similarity = 0.0

        for visitor_id, visitor_data in self.known_visitors.items():
            similarity = self.calculate_feature_similarity(features, visitor_data['features'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = visitor_id

        return best_match_id, best_similarity

    def _create_new_visitor(self, features, face_image, track_id):
        """Создание нового посетителя"""
        cursor = self.conn.cursor()
        now = datetime.datetime.now()

        visitor_id = None
        try:
            # Сохраняем фото
            photo_path = self.save_visitor_photo(face_image, "temp")

            # Создаем запись в базе
            features_blob = features.astype(np.float32).tobytes()
            cursor.execute(
                """INSERT INTO visitors (first_seen, last_seen, visit_count, 
                   last_updated, photo_path, track_features) 
                   VALUES (?, ?, 1, ?, ?, ?)""",
                (now, now, now, photo_path, features_blob)
            )
            visitor_id = cursor.lastrowid

            # Обновляем путь к фото с правильным ID
            final_photo_path = self.save_visitor_photo(face_image, visitor_id)
            cursor.execute(
                "UPDATE visitors SET photo_path = ? WHERE id = ?",
                (final_photo_path, visitor_id)
            )

            # Сохраняем в памяти
            self.known_visitors[visitor_id] = {
                'features': features,
                'photo_path': final_photo_path,
                'first_seen': now
            }

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
        if current_time - self.last_log_time >= 3.0:
            logger.info(f"📊 СТАТИСТИКА: Всего в базе: {len(self.known_visitors)}, "
                        f"Активных треков: {len(self.face_tracks)}, "
                        f"В галерее: {len(self.current_visitors_gallery)}, "
                        f"Новых за сессию: {self.recognition_stats['new_visitors']}, "
                        f"Известных: {self.recognition_stats['known_visitors']}, "
                        f"Отфильтровано: {self.recognition_stats['filtered_detections']}")
            self.last_log_time = current_time

    def process_frame(self, frame):
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
            return frame, 0, 0, 0

        # Детекция лиц
        detected_faces = self.detect_faces_mediapipe(frame)

        processed_faces = []
        filtered_count = 0

        if detected_faces:
            self.recognition_stats['total_detections'] += len(detected_faces)
            logger.info(f"👥 ОБНАРУЖЕНО ОБЪЕКТОВ: {len(detected_faces)}")

            for face_data in detected_faces:
                x, y, w, h = face_data['coords']

                face_img = frame[y:y + h, x:x + w]

                # Проверка характеристик человека
                if not self.validate_human_features(face_img):
                    filtered_count += 1
                    logger.debug("❌ Отфильтрован не-человеческий объект")
                    continue

                features = self.extract_simple_features(face_img)
                if features is not None:
                    processed_faces.append({
                        'coords': (x, y, w, h),
                        'features': features,
                        'confidence': face_data['confidence'],
                        'face_image': face_img,
                        'status': 'detected'
                    })

        self.recognition_stats['filtered_detections'] += filtered_count

        # Обновляем трекинг
        tracked_faces = self.update_face_tracking(processed_faces, current_time)

        # Отрисовка результатов
        processed_frame = frame.copy()
        processed_count = 0

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
                if 'similarity' in face_data:
                    cv2.putText(processed_frame, f'Sim: {face_data["similarity"]:.2f}',
                                (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            processed_count += 1

        self.last_processing_time = current_time
        self.log_recognition_stats()

        return processed_frame, len(detected_faces), processed_count, filtered_count

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

    def start_analysis(self, rtsp_url):
        """Запуск анализа"""
        logger.info("🚀 Запуск облегченной версии с MediaPipe...")

        cap = self.setup_rtsp_camera(rtsp_url)
        if not cap.isOpened():
            return

        logger.info("✅ Анализ запущен")

        window_name = 'Trassir Analytics - LIGHTWEIGHT (MediaPipe only)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("📡 Потеряно соединение с камерой...")
                    time.sleep(2)
                    continue

                processed_frame, detected, processed, filtered = self.process_frame(frame)
                display_frame = self.resize_frame_for_display(processed_frame, target_width=1280)
                display_with_gallery = self.create_gallery_display(display_frame)

                # Статистика на экране
                stats_text = [
                    f"LIGHTWEIGHT ANALYTICS (MediaPipe)",
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
            cap.release()
            cv2.destroyAllWindows()
            self.conn.close()

            logger.info(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
            logger.info(f"   Всего посетителей: {len(self.known_visitors)}")
            logger.info(f"   Новых создано: {self.recognition_stats['new_visitors']}")
            logger.info(f"   Известных обработано: {self.recognition_stats['known_visitors']}")
            logger.info(f"   Отфильтровано объектов: {self.recognition_stats['filtered_detections']}")
            logger.info("✅ Анализ завершен")


def main():
    """Основная функция"""
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    counter = LightweightTrassirCounter(
        processing_interval=1.0,
        tracking_threshold=0.6
    )

    try:
        counter.start_analysis(RTSP_URL)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()