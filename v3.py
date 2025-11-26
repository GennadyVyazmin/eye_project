# video_analytics_trassir_deepsort_fixed.py
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

# Попробуем разные варианты импорта DeepSORT
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort

    DEEPSORT_AVAILABLE = True
    logger.info("✅ DeepSORT доступен (вариант 1)")
except ImportError:
    try:
        from deep_sort_realtime import DeepSort

        DEEPSORT_AVAILABLE = True
        logger.info("✅ DeepSORT доступен (вариант 2)")
    except ImportError as e:
        DEEPSORT_AVAILABLE = False
        logger.warning(f"❌ DeepSORT не доступен: {e}")
        logger.info("⚠️  Используем базовый трекинг")

# Альтернативный вариант - проверка через pkg_resources
try:
    import pkg_resources

    pkg_resources.get_distribution("deep-sort-realtime")
    DEEPSORT_AVAILABLE = True
    logger.info("✅ DeepSORT доступен (через pkg_resources)")
except:
    DEEPSORT_AVAILABLE = False
    logger.warning("❌ DeepSORT не найден через pkg_resources")


class AdvancedTrassirCounter:
    def __init__(self, processing_interval=1.0, similarity_threshold=0.55, tracking_threshold=0.45):
        """
        Продвинутая версия с DeepSORT для трекинга по внешности
        """
        self.conn = sqlite3.connect('visitors_trassir_advanced.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
        self.similarity_threshold = similarity_threshold
        self.tracking_threshold = tracking_threshold

        # Инициализация DeepSORT
        self.deepsort = None
        if DEEPSORT_AVAILABLE:
            try:
                # Пробуем разные параметры инициализации
                self.deepsort = DeepSort(
                    max_age=50,  # Увеличили время жизни трека
                    n_init=3,  # Количество кадров для инициализации
                    max_cosine_distance=0.2,  # Более строгий порог
                    nn_budget=50,  # Уменьшили бюджет для производительности
                    override_track_class=None
                )
                logger.info("🚀 DeepSORT успешно инициализирован")
            except Exception as e:
                logger.error(f"❌ Ошибка инициализации DeepSORT: {e}")
                logger.info("🔄 Пробуем альтернативную инициализацию...")
                try:
                    # Альтернативные параметры
                    self.deepsort = DeepSort(max_age=30)
                    logger.info("✅ DeepSORT инициализирован с альтернативными параметрами")
                except Exception as e2:
                    logger.error(f"❌ Полная ошибка DeepSORT: {e2}")
                    self.deepsort = None
        else:
            logger.info("ℹ️  Используем базовый трекинг без DeepSORT")

        # Цвета для индикации статусов
        self.COLORS = {
            'detected': (0, 255, 0),  # Зеленый - лицо обнаружено
            'tracking': (255, 255, 0),  # Желтый - создан трек
            'known': (0, 255, 255),  # Голубой - известный пользователь
            'new': (0, 0, 255),  # Красный - новый пользователь в БД
            'analyzing': (255, 165, 0),  # Оранжевый - анализ в процессе
            'deepsort': (255, 0, 255)  # Пурпурный - трекинг DeepSORT
        }

        # Папки для хранения фото
        self.photos_dir = "visitor_photos_advanced"
        self.current_session_dir = "current_session"
        self._create_directories()

        # Трекинг состояния
        self.last_processing_time = 0
        self.known_visitors_cache = {}
        self.frame_count = 0

        # Комбинированная система трекинга
        self.face_tracks = {}
        self.deepsort_tracks = {}
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
            'deepsort_matches': 0,
            'face_only_matches': 0
        }
        self.last_log_time = time.time()

        # Очередь для обработки
        self.frame_queue = Queue(maxsize=1)
        self.results_queue = Queue()

        # Детектор лиц
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
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

        logger.info(f"🎯 Система инициализирована. DeepSORT: {'АКТИВЕН' if self.deepsort else 'НЕДОСТУПЕН'}")

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
                appearance_features BLOB
            )
        ''')
        self.conn.commit()

    def _load_known_visitors(self):
        """Загрузка известных посетителей"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, face_embedding, photo_path, appearance_features FROM visitors")
        visitors = cursor.fetchall()

        self.known_visitors_cache.clear()
        for visitor_id, embedding_blob, photo_path, appearance_blob in visitors:
            if embedding_blob:
                try:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    appearance_features = np.frombuffer(appearance_blob, dtype=np.float32) if appearance_blob else None

                    self.known_visitors_cache[visitor_id] = {
                        'embedding': embedding,
                        'photo_path': photo_path,
                        'appearance_features': appearance_features
                    }
                except Exception as e:
                    logger.warning(f"Ошибка загрузки посетителя {visitor_id}: {e}")

        logger.info(f"📊 Загружено посетителей: {len(self.known_visitors_cache)}")

    def process_with_deepsort(self, frame, faces):
        """Обработка кадра с DeepSORT"""
        if self.deepsort is None:
            return []

        try:
            # Подготавливаем детекции для DeepSORT
            detections = []
            for (x, y, w, h) in faces:
                confidence = 0.9  # Высокая уверенность для лиц
                # DeepSORT ожидает формат [x1, y1, x2, y2, confidence]
                bbox = [x, y, x + w, y + h]
                detections.append((bbox, confidence, None))

            # Обновляем треки DeepSORT
            tracks = self.deepsort.update_tracks(detections, frame=frame)

            deepsort_results = []
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr()  # [x1, y1, x2, y2]

                # Конвертируем в формат [x, y, w, h]
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2] - bbox[0])
                h = int(bbox[3] - bbox[1])

                deepsort_results.append({
                    'track_id': track_id,
                    'coords': (x, y, w, h),
                    'type': 'deepsort'
                })

                # Обновляем кэш треков DeepSORT
                self.deepsort_tracks[track_id] = {
                    'coords': (x, y, w, h),
                    'last_seen': time.time()
                }

            if deepsort_results:
                logger.debug(f"🔍 DeepSORT отслеживает {len(deepsort_results)} объектов")

            return deepsort_results

        except Exception as e:
            logger.error(f"❌ Ошибка DeepSORT обработки: {e}")
            return []

    def extract_appearance_features(self, full_body_image):
        """Извлечение признаков внешности (упрощенная версия)"""
        try:
            # Простые гистограммы цвета как базовые признаки
            hsv = cv2.cvtColor(full_body_image, cv2.COLOR_BGR2HSV)

            # Гистограммы по каналам
            hist_hue = cv2.calcHist([hsv], [0], None, [8], [0, 180])
            hist_sat = cv2.calcHist([hsv], [1], None, [4], [0, 256])
            hist_val = cv2.calcHist([hsv], [2], None, [4], [0, 256])

            # Нормализация и объединение
            hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()
            hist_sat = cv2.normalize(hist_sat, hist_sat).flatten()
            hist_val = cv2.normalize(hist_val, hist_val).flatten()

            appearance_features = np.concatenate([hist_hue, hist_sat, hist_val])
            return appearance_features.astype(np.float32)

        except Exception as e:
            logger.debug(f"Ошибка извлечения признаков внешности: {e}")
            return None

    def calculate_appearance_similarity(self, features1, features2):
        """Расчет схожести по внешним признакам"""
        if features1 is None or features2 is None:
            return 0.0

        try:
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(features1, features2) / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.debug(f"Ошибка расчета схожести внешности: {e}")
            return 0.0

    def match_face_with_deepsort(self, face_coords, face_embedding):
        """Сопоставление лица с треками DeepSORT"""
        if not self.deepsort_tracks:
            return None, 0.0

        face_x, face_y, face_w, face_h = face_coords
        face_center = (face_x + face_w // 2, face_y + face_h // 2)

        best_track_id = None
        best_similarity = 0.0

        for track_id, track_data in list(self.deepsort_tracks.items()):
            # Проверяем возраст трека
            if time.time() - track_data['last_seen'] > self.track_max_age:
                del self.deepsort_tracks[track_id]
                continue

            track_x, track_y, track_w, track_h = track_data['coords']
            track_center = (track_x + track_w // 2, track_y + track_h // 2)

            # Расчет расстояния между центрами
            distance = np.sqrt((face_center[0] - track_center[0]) ** 2 +
                               (face_center[1] - track_center[1]) ** 2)

            # Нормализованное расстояние (чем меньше, тем лучше)
            max_distance = max(face_w, face_h, track_w, track_h)
            if max_distance > 0:
                normalized_distance = 1.0 - min(1.0, distance / max_distance)
            else:
                normalized_distance = 0.0

            # Если центры достаточно близко, считаем это совпадением
            if normalized_distance > 0.6:  # Порог близости
                if normalized_distance > best_similarity:
                    best_similarity = normalized_distance
                    best_track_id = track_id

        return best_track_id, best_similarity

    def combined_similarity_score(self, face_similarity, appearance_similarity, spatial_similarity):
        """Комбинированная оценка схожести"""
        # Весовые коэффициенты
        face_weight = 0.7  # Лицо - самый важный признак
        appearance_weight = 0.2  # Внешность (одежда)
        spatial_weight = 0.1  # Пространственное положение

        total_score = (face_similarity * face_weight +
                       appearance_similarity * appearance_weight +
                       spatial_similarity * spatial_weight)

        return total_score

    def find_best_combined_match(self, embedding, appearance_features, coords):
        """Поиск лучшего совпадения с комбинированными признаками"""
        if embedding is None:
            return None, 0.0

        best_match_id = None
        best_combined_similarity = 0.0

        for visitor_id, visitor_data in self.known_visitors_cache.items():
            # Схожесть лиц
            face_similarity = self.calculate_similarity(embedding, visitor_data['embedding'])

            # Схожесть внешности
            appearance_similarity = 0.0
            if appearance_features is not None and visitor_data['appearance_features'] is not None:
                appearance_similarity = self.calculate_appearance_similarity(
                    appearance_features, visitor_data['appearance_features']
                )

            # Пространственная схожесть
            spatial_similarity = 0.5  # Нейтральное значение

            # Комбинированная оценка
            combined_similarity = self.combined_similarity_score(
                face_similarity, appearance_similarity, spatial_similarity
            )

            if combined_similarity > best_combined_similarity:
                best_combined_similarity = combined_similarity
                best_match_id = visitor_id

        # Логируем тип совпадения
        if best_match_id and best_combined_similarity > self.similarity_threshold:
            face_only_similarity = self.calculate_similarity(embedding,
                                                             self.known_visitors_cache[best_match_id]['embedding'])
            if face_only_similarity < self.similarity_threshold:
                self.recognition_stats['deepsort_matches'] += 1
                logger.info(f"🎯 DeepSORT помог опознать пользователя {best_match_id}")
            else:
                self.recognition_stats['face_only_matches'] += 1

        return best_match_id, best_combined_similarity

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

    def get_fast_embedding(self, face_image):
        """Быстрое получение эмбеддинга"""
        try:
            face_resized = cv2.resize(face_image, (112, 112))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            face_normalized = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

            result = DeepFace.represent(
                face_normalized,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='skip',
                align=True
            )

            embedding = np.array(result[0]['embedding'], dtype=np.float32)

            if np.all(embedding == 0) or np.linalg.norm(embedding) < 0.1:
                return None

            return embedding

        except Exception as e:
            logger.debug(f"Ошибка получения эмбеддинга: {e}")
            return None

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

            logger.info(f"📸 Сохранено фото посетителя {visitor_id}: {filename}")
            return filepath

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения фото для посетителя {visitor_id}: {e}")
            return None

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
            'deepsort': 'DEEPSORT'
        }
        return status_texts.get(status, 'UNKNOWN')

    def get_color_by_status(self, status):
        """Получение цвета по статусу"""
        return self.COLORS.get(status, (255, 255, 255))

    def create_gallery_display(self, main_frame):
        """Создание галереи посетителей справа от основного изображения"""
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
        """Умное изменение размера кадра для отображения"""
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
            logger.info(f"📈 СТАТИСТИКА: Всего в базе: {len(self.known_visitors_cache)}, "
                        f"Активных треков: {len(self.face_tracks)}, "
                        f"В галерее: {len(self.current_visitors_gallery)}, "
                        f"Новых за сессию: {self.recognition_stats['new_visitors']}")

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
        """Тяжелые операции обработки с DeepSORT"""
        try:
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame_small = cv2.resize(frame, (new_width, new_height))
            else:
                frame_small = frame

            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                maxSize=(300, 300),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            deepsort_tracks = self.process_with_deepsort(frame, faces)

            processed_faces = []
            if len(faces) > 0:
                logger.debug(f"👥 Обнаружено лиц: {len(faces)}, DeepSORT треков: {len(deepsort_tracks)}")

                for (x, y, w, h) in faces:
                    scale_x = width / frame_small.shape[1]
                    scale_y = height / frame_small.shape[0]

                    x_orig = int(x * scale_x)
                    y_orig = int(y * scale_y)
                    w_orig = int(w * scale_x)
                    h_orig = int(h * scale_y)

                    if 50 <= w_orig <= 400 and 50 <= h_orig <= 400:
                        face_img = frame[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]

                        embedding = self.get_fast_embedding(face_img)
                        if embedding is not None:
                            appearance_features = self.extract_appearance_features(
                                frame[max(0, y_orig - 50):min(height, y_orig + h_orig + 50),
                                max(0, x_orig - 50):min(width, x_orig + w_orig + 50)]
                            )

                            visitor_id, similarity = self.find_best_combined_match(
                                embedding, appearance_features, (x_orig, y_orig, w_orig, h_orig)
                            )

                            deepsort_track_id, deepsort_similarity = self.match_face_with_deepsort(
                                (x_orig, y_orig, w_orig, h_orig), embedding
                            )

                            status = 'detected'
                            if deepsort_track_id:
                                status = 'deepsort'
                                logger.debug(f"🔗 Лицо связано с DeepSORT треком {deepsort_track_id}")

                            processed_faces.append({
                                'coords': (x_orig, y_orig, w_orig, h_orig),
                                'embedding': embedding,
                                'similarity': similarity,
                                'visitor_id': visitor_id,
                                'appearance_features': appearance_features,
                                'deepsort_track_id': deepsort_track_id,
                                'status': status,
                                'face_image': face_img
                            })

            return {
                'faces': processed_faces,
                'processed_count': len(processed_faces),
                'detected_count': len(faces),
                'deepsort_tracks': len(deepsort_tracks),
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Ошибка в фоновой обработке: {e}")
            return {'faces': [], 'processed_count': 0, 'detected_count': 0}

    def setup_rtsp_camera(self, rtsp_url):
        """Настройка RTSP"""
        logger.info(f"📡 Подключение к камере: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

        for _ in range(5):
            cap.read()

        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                logger.info(f"✅ Камера подключена. Разрешение: {test_frame.shape[1]}x{test_frame.shape[0]}")
        else:
            logger.error("❌ Не удалось подключиться к камере")

        return cap

    def process_frame_realtime(self, frame):
        """Обработка кадра в реальном времени"""
        current_time = time.time()

        self.fps_frame_count += 1
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_frame_count / (current_time - self.fps_start_time)
            self.fps_frame_count = 0
            self.fps_start_time = current_time

        if current_time - self.last_processing_time < self.processing_interval:
            try:
                result, frame_time = self.results_queue.get_nowait()
                return self._apply_processing_result(frame, result, current_time)
            except:
                return frame, 0, 0

        if self.frame_queue.empty():
            self.frame_queue.put((frame.copy(), current_time))

        self.last_processing_time = current_time

        try:
            result, frame_time = self.results_queue.get_nowait()
            return self._apply_processing_result(frame, result, current_time)
        except:
            return frame, 0, 0

    def _apply_processing_result(self, frame, result, current_time):
        """Применяет результаты обработки"""
        processed_frame = frame.copy()
        processed_count = 0

        for face_data in result['faces']:
            x, y, w, h = face_data['coords']
            status = face_data.get('status', 'detected')
            visitor_id = face_data.get('visitor_id')

            color = self.get_color_by_status(status)
            status_text = self.get_status_text(status)

            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(processed_frame, f'{status_text}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if visitor_id:
                cv2.putText(processed_frame, f'ID: {visitor_id}', (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                self.update_visitor_gallery(visitor_id, face_data['face_image'], status)

            processed_count += 1

        self.log_recognition_stats()

        return processed_frame, result['detected_count'], processed_count

    def start_analysis(self, rtsp_url):
        """Запуск анализа с DeepSORT"""
        logger.info("🚀 Запуск продвинутой версии с DeepSORT...")

        cap = self.setup_rtsp_camera(rtsp_url)
        if not cap.isOpened():
            return

        self.start_processing_thread()

        if self.deepsort:
            logger.info("✅ DeepSORT АКТИВЕН - используется комбинированная идентификация")
        else:
            logger.info("ℹ️  DeepSORT НЕДОСТУПЕН - используется базовая идентификация по лицам")

        window_name = 'Trassir Analytics - DEEPSORT'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("📡 Потеряно соединение...")
                    time.sleep(2)
                    continue

                processed_frame, detected, processed = self.process_frame_realtime(frame)
                display_frame = self.resize_frame_for_display(processed_frame, target_width=1280)
                display_with_gallery = self.create_gallery_display(display_frame)

                deepsort_status = "ACTIVE" if self.deepsort else "DISABLED"
                stats_text = [
                    f"ADVANCED ANALYTICS - DeepSORT: {deepsort_status}",
                    f"Faces detected: {detected}",
                    f"Visitors processed: {processed}",
                    f"DeepSORT assists: {self.recognition_stats['deepsort_matches']}",
                    f"Face matches: {self.recognition_stats['face_only_matches']}",
                    f"FPS: {self.current_fps:.1f}",
                    f"Press 'q' to quit"
                ]

                overlay = display_with_gallery.copy()
                cv2.rectangle(overlay, (0, 0), (500, 200), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_with_gallery, 0.3, 0, display_with_gallery)

                for i, text in enumerate(stats_text):
                    cv2.putText(display_with_gallery, text, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(window_name, display_with_gallery)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("⏹️ Остановка...")
        finally:
            self.stop_processing = True
            if self.processing_thread:
                self.processing_thread.join(timeout=2.0)
            cap.release()
            cv2.destroyAllWindows()
            self.conn.close()

            logger.info(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
            logger.info(f"   Всего посетителей: {len(self.known_visitors_cache)}")
            if self.deepsort:
                logger.info(f"   Совпадений с помощью DeepSORT: {self.recognition_stats['deepsort_matches']}")
            logger.info(f"   Совпадений по лицу: {self.recognition_stats['face_only_matches']}")
            logger.info("✅ Анализ завершен")


def main():
    """Основная функция"""
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    counter = AdvancedTrassirCounter(
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