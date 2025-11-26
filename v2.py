# video_analytics_trassir_with_gallery.py
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


class OptimizedTrassirCounter:
    def __init__(self, processing_interval=1.5, similarity_threshold=0.55, tracking_threshold=0.45):
        """
        Версия с созданием фото и галереей посетителей
        """
        self.conn = sqlite3.connect('visitors_trassir_opt_v2.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
        self.similarity_threshold = similarity_threshold
        self.tracking_threshold = tracking_threshold

        # Цвета для индикации статусов
        self.COLORS = {
            'detected': (0, 255, 0),  # Зеленый - лицо обнаружено
            'tracking': (255, 255, 0),  # Желтый - создан трек
            'known': (0, 255, 0),  # Зеленый - известный пользователь
            'new': (0, 0, 255),  # Красный - новый пользователь в БД
            'analyzing': (255, 165, 0)  # Оранжевый - анализ в процессе
        }

        # Папки для хранения фото
        self.photos_dir = "visitor_photos"
        self.current_session_dir = "current_session"
        self._create_directories()

        # Трекинг состояния
        self.last_processing_time = 0
        self.known_visitors_cache = {}
        self.frame_count = 0

        # Система трекинга лиц между кадрами
        self.face_tracks = {}
        self.next_track_id = 1
        self.track_max_age = 3.0

        # Галерея текущих посетителей в кадре
        self.current_visitors_gallery = {}  # {visitor_id: {'photo': image, 'last_seen': timestamp}}
        self.gallery_max_size = 8  # Максимальное количество фото в галерее
        self.photo_size = (120, 160)  # Размер фото в галерее

        # Статистика для логирования
        self.recognition_stats = {
            'total_detections': 0,
            'new_visitors': 0,
            'known_visitors': 0,
            'duplicates_prevented': 0
        }
        self.last_log_time = time.time()

        # Очередь для асинхронной обработки
        self.frame_queue = Queue(maxsize=1)
        self.results_queue = Queue()

        # Загрузка детектора лиц
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

        logger.info(f"Улучшенная инициализация с галереей посетителей")

    def _create_directories(self):
        """Создание папок для хранения фото"""
        os.makedirs(self.photos_dir, exist_ok=True)
        os.makedirs(os.path.join(self.photos_dir, self.current_session_dir), exist_ok=True)
        logger.info(f"📁 Созданы папки для фото: {self.photos_dir}")

    def _init_database(self):
        """Инициализация базы данных с полем для пути к фото"""
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
        """Загрузка известных посетителей в кэш"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, face_embedding, photo_path FROM visitors")
        visitors = cursor.fetchall()

        self.known_visitors_cache.clear()
        for visitor_id, embedding_blob, photo_path in visitors:
            if embedding_blob:
                try:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    self.known_visitors_cache[visitor_id] = {
                        'embedding': embedding,
                        'photo_path': photo_path
                    }
                except Exception as e:
                    logger.warning(f"Ошибка загрузки посетителя {visitor_id}: {e}")

        logger.info(f"📊 Загружено посетителей из базы: {len(self.known_visitors_cache)}")

    def save_visitor_photo(self, face_image, visitor_id):
        """Сохранение фото посетителя"""
        try:
            # Создаем чистую версию фото (без рамок и текста)
            photo_clean = face_image.copy()

            # Увеличиваем немного размер для лучшего качества
            height, width = photo_clean.shape[:2]
            if width < 200:
                scale = 200 / width
                new_width = 200
                new_height = int(height * scale)
                photo_clean = cv2.resize(photo_clean, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # Сохраняем фото
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
            # Подготавливаем фото для галереи
            gallery_photo = face_image.copy()

            # Добавляем рамку статуса
            border_color = self.COLORS.get(status, (255, 255, 255))
            gallery_photo = cv2.copyMakeBorder(
                gallery_photo, 5, 25, 5, 5, cv2.BORDER_CONSTANT, value=border_color
            )

            # Добавляем текст с ID и статусом
            status_text = self.get_status_text(status)
            cv2.putText(gallery_photo, f"ID: {visitor_id}", (10, gallery_photo.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)

            # Ресайз для галереи
            gallery_photo = cv2.resize(gallery_photo, self.photo_size, interpolation=cv2.INTER_AREA)

            # Обновляем галерею
            self.current_visitors_gallery[visitor_id] = {
                'photo': gallery_photo,
                'last_seen': time.time(),
                'status': status
            }

            # Ограничиваем размер галереи
            if len(self.current_visitors_gallery) > self.gallery_max_size:
                # Удаляем самого старого посетителя
                oldest_visitor = min(self.current_visitors_gallery.keys(),
                                     key=lambda x: self.current_visitors_gallery[x]['last_seen'])
                del self.current_visitors_gallery[oldest_visitor]

        except Exception as e:
            logger.error(f"Ошибка обновления галереи: {e}")

    def create_gallery_display(self, main_frame):
        """Создание галереи посетителей справа от основного изображения"""
        try:
            main_height, main_width = main_frame.shape[:2]

            # Создаем панель для галереи
            gallery_width = 300
            gallery_panel = np.zeros((main_height, gallery_width, 3), dtype=np.uint8)

            # Заголовок галереи
            cv2.putText(gallery_panel, "CURRENT VISITORS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(gallery_panel, f"Total: {len(self.current_visitors_gallery)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Размещаем фото в галерее
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

                    # Проверяем чтобы фото поместилось
                    if y + photo_height < main_height and x + photo_width < gallery_width:
                        gallery_panel[y:y + photo_height, x:x + photo_width] = visitor_data['photo']

                        # Добавляем маленький индикатор статуса
                        status_color = self.COLORS.get(visitor_data['status'], (255, 255, 255))
                        cv2.circle(gallery_panel, (x + 10, y + 10), 5, status_color, -1)
            else:
                # Сообщение если галерея пуста
                cv2.putText(gallery_panel, "No visitors", (50, main_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
                cv2.putText(gallery_panel, "in frame", (60, main_height // 2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)

            # Объединяем основное изображение с галереей
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
        """Тяжелые операции обработки с улучшенным трекингом"""
        try:
            # Уменьшаем разрешение для обработки
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame_small = cv2.resize(frame, (new_width, new_height))
            else:
                frame_small = frame

            # Детекция лиц
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(60, 60),
                maxSize=(300, 300),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            processed_faces = []
            if len(faces) > 0:
                logger.info(f"👥 Обнаружено лиц в кадре: {len(faces)}")

                for (x, y, w, h) in faces:
                    # Масштабируем координаты обратно
                    scale_x = width / frame_small.shape[1]
                    scale_y = height / frame_small.shape[0]

                    x_orig = int(x * scale_x)
                    y_orig = int(y * scale_y)
                    w_orig = int(w * scale_x)
                    h_orig = int(h * scale_y)

                    if 60 <= w_orig <= 400 and 60 <= h_orig <= 400:
                        face_img = frame[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]

                        embedding = self.get_fast_embedding(face_img)
                        if embedding is not None:
                            visitor_id, similarity = self.find_best_match(embedding)

                            processed_faces.append({
                                'coords': (x_orig, y_orig, w_orig, h_orig),
                                'embedding': embedding,
                                'similarity': similarity,
                                'visitor_id': visitor_id,
                                'is_confirmed': similarity > self.similarity_threshold,
                                'status': 'detected',
                                'face_image': face_img  # Сохраняем изображение лица
                            })

            return {
                'faces': processed_faces,
                'processed_count': len(processed_faces),
                'detected_count': len(faces),
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Ошибка в фоновой обработке: {e}")
            return {'faces': [], 'processed_count': 0, 'detected_count': 0}

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

    def calculate_similarity(self, embedding1, embedding2):
        """Безопасный расчет схожести"""
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

        for visitor_id, visitor_data in self.known_visitors_cache.items():
            similarity = self.calculate_similarity(embedding, visitor_data['embedding'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = visitor_id

        if best_similarity > self.tracking_threshold:
            return best_match_id, best_similarity

        return None, best_similarity

    def update_face_tracking(self, current_faces, timestamp):
        """Обновление трекинга лиц между кадрами"""
        updated_faces = []

        for face_data in current_faces:
            embedding = face_data['embedding']
            coords = face_data['coords']
            face_image = face_data['face_image']

            best_track_id = None
            best_similarity = 0.0

            for track_id, track_data in list(self.face_tracks.items()):
                if timestamp - track_data['last_seen'] > self.track_max_age:
                    del self.face_tracks[track_id]
                    continue

                similarity = self.calculate_similarity(embedding, track_data['embedding'])
                if similarity > best_similarity and similarity > self.tracking_threshold:
                    best_similarity = similarity
                    best_track_id = track_id

            if best_track_id is not None:
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
                track_id = self.next_track_id
                self.next_track_id += 1
                self.face_tracks[track_id] = {
                    'embedding': embedding,
                    'last_seen': timestamp,
                    'coords': coords,
                    'face_image': face_image,
                    'visitor_id': None,
                    'created_at': timestamp,
                    'status': 'tracking'
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

        if track_data['visitor_id']:
            visitor_id = track_data['visitor_id']
            if visitor_id in self.known_visitors_cache:
                old_embedding = self.known_visitors_cache[visitor_id]['embedding']
                new_embedding = 0.7 * old_embedding + 0.3 * embedding
                self.known_visitors_cache[visitor_id]['embedding'] = new_embedding / np.linalg.norm(new_embedding)

            self.recognition_stats['known_visitors'] += 1
            face_data['status'] = 'known'

            # Обновляем галерею для известного пользователя
            self.update_visitor_gallery(visitor_id, face_image, 'known')

            logger.debug(f"♻️  Подтвержден известный посетитель {visitor_id}")
            return visitor_id

        visitor_id, similarity = self.find_best_match(embedding)

        if similarity > self.similarity_threshold:
            track_data['visitor_id'] = visitor_id
            track_data['confirmed_at'] = timestamp
            self.recognition_stats['known_visitors'] += 1
            face_data['status'] = 'known'

            # Обновляем галерею
            self.update_visitor_gallery(visitor_id, face_image, 'known')

            logger.info(f"👤 ОПОЗНАН известный посетитель {visitor_id}, схожесть: {similarity:.3f}")
            return visitor_id
        else:
            track_duration = timestamp - track_data['created_at']
            if track_duration > 2.0:
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
        """Создание нового посетителя с фото"""
        cursor = self.conn.cursor()
        now = datetime.datetime.now()

        # Сохраняем фото
        visitor_id = None
        try:
            # Сначала создаем запись чтобы получить ID
            embedding_blob = embedding.astype(np.float32).tobytes()
            cursor.execute(
                """INSERT INTO visitors (face_embedding, first_seen, last_seen, 
                   visit_count, last_updated, confirmed_count, photo_path) 
                   VALUES (?, ?, ?, 1, ?, 1, ?)""",
                (embedding_blob, now, now, now, '')
            )
            visitor_id = cursor.lastrowid

            # Сохраняем фото
            photo_path = self.save_visitor_photo(face_image, visitor_id)

            # Обновляем запись с путем к фото
            cursor.execute(
                "UPDATE visitors SET photo_path = ? WHERE id = ?",
                (photo_path, visitor_id)
            )

            self.known_visitors_cache[visitor_id] = {
                'embedding': embedding,
                'photo_path': photo_path
            }
            self.conn.commit()

        except Exception as e:
            logger.error(f"❌ Ошибка создания посетителя: {e}")
            self.conn.rollback()
            return None

        if track_id in self.face_tracks:
            self.face_tracks[track_id]['visitor_id'] = visitor_id

        return visitor_id

    def save_visitor_visit(self, visitor_id, embedding):
        """Сохранение визита посетителя"""
        if visitor_id is None:
            return

        cursor = self.conn.cursor()
        now = datetime.datetime.now()

        cursor.execute(
            "UPDATE visitors SET last_seen = ?, visit_count = visit_count + 1, last_updated = ? WHERE id = ?",
            (now, now, visitor_id)
        )
        self.conn.commit()

    def get_color_by_status(self, status):
        """Получение цвета по статусу"""
        return self.COLORS.get(status, (255, 255, 255))

    def get_status_text(self, status):
        """Получение текста по статусу"""
        status_texts = {
            'detected': 'DETECTED',
            'tracking': 'TRACKING',
            'analyzing': 'ANALYZING',
            'known': 'KNOWN',
            'new': 'NEW USER'
        }
        return status_texts.get(status, 'UNKNOWN')

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
        """Обработка кадра с улучшенным трекингом"""
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
        """Применяет результаты обработки с трекингом и цветовой индикацией"""
        processed_frame = frame.copy()
        processed_count = 0

        tracked_faces = self.update_face_tracking(result['faces'], current_time)

        for face_data in tracked_faces:
            visitor_id = self.confirm_visitor_identity(face_data['track_id'], face_data)

            if visitor_id:
                self.save_visitor_visit(visitor_id, face_data['embedding'])

            x, y, w, h = face_data['coords']
            status = face_data.get('status', 'detected')

            # Получаем цвет и текст по статусу
            color = self.get_color_by_status(status)
            status_text = self.get_status_text(status)

            # Отрисовка рамки с цветом статуса
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 3)

            # Текст статуса
            cv2.putText(processed_frame, f'{status_text}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # ID пользователя (если есть)
            if visitor_id:
                cv2.putText(processed_frame, f'ID: {visitor_id}', (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            processed_count += 1

        self.log_recognition_stats()

        return processed_frame, result['detected_count'], processed_count

    def start_analysis(self, rtsp_url):
        """Запуск анализа с галереей посетителей"""
        logger.info("🚀 Запуск версии с галереей посетителей...")

        cap = self.setup_rtsp_camera(rtsp_url)
        if not cap.isOpened():
            return

        self.start_processing_thread()

        logger.info("✅ Анализ с галереей запущен")

        # Создаем окно
        window_name = 'Trassir Analytics - VISITOR GALLERY'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)  # Увеличили окно для галереи

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("📡 Потеряно соединение с камерой...")
                    time.sleep(2)
                    continue

                processed_frame, detected, processed = self.process_frame_realtime(frame)

                display_frame = self.resize_frame_for_display(processed_frame, target_width=1280)

                # Добавляем галерею
                display_with_gallery = self.create_gallery_display(display_frame)

                # Статистика на экране
                stats_text = [
                    f"VISITOR ANALYTICS WITH GALLERY",
                    f"Detected: {detected}",
                    f"Processed: {processed}",
                    f"Total in DB: {len(self.known_visitors_cache)}",
                    f"In Gallery: {len(self.current_visitors_gallery)}",
                    f"FPS: {self.current_fps:.1f}",
                    f"Press 'q' to quit"
                ]

                # Фон для текста
                overlay = display_with_gallery.copy()
                cv2.rectangle(overlay, (0, 0), (500, 180), (0, 0, 0), -1)
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
            logger.info(f"   Всего уникальных посетителей: {len(self.known_visitors_cache)}")
            logger.info(f"   Новых создано за сессию: {self.recognition_stats['new_visitors']}")
            logger.info(f"   Фото сохранено в: {self.photos_dir}")
            logger.info("✅ Анализ завершен")

    def cleanup_database(self):
        """Очистка базы данных и фото"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM visitors")
        self.conn.commit()
        self.known_visitors_cache.clear()
        self.face_tracks.clear()
        self.current_visitors_gallery.clear()

        # Очищаем папку текущей сессии
        session_dir = os.path.join(self.photos_dir, self.current_session_dir)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            os.makedirs(session_dir)

        logger.info("🗑️ База данных и фото сессии очищены")


def main():
    """Основная функция"""
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    counter = OptimizedTrassirCounter(
        processing_interval=1.5,
        similarity_threshold=0.55,
        tracking_threshold=0.45
    )

    # counter.cleanup_database()

    try:
        counter.start_analysis(RTSP_URL)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()