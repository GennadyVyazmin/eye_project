# video_analytics_trassir_optimized_fixed_gui.py
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

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedTrassirCounter:
    def __init__(self, processing_interval=1.5, similarity_threshold=0.55, tracking_threshold=0.45):
        """
        Версия с фиксированным нормальным размером GUI
        """
        self.conn = sqlite3.connect('visitors_trassir_opt_v2.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
        self.similarity_threshold = similarity_threshold
        self.tracking_threshold = tracking_threshold

        # Трекинг состояния
        self.last_processing_time = 0
        self.known_visitors_cache = {}
        self.frame_count = 0

        # Система трекинга лиц между кадрами
        self.face_tracks = {}
        self.next_track_id = 1
        self.track_max_age = 3.0

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

        logger.info(f"Улучшенная инициализация с нормальным GUI")

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
                confirmed_count INTEGER DEFAULT 1
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
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    self.known_visitors_cache[visitor_id] = embedding
                except Exception as e:
                    logger.warning(f"Ошибка загрузки посетителя {visitor_id}: {e}")

        logger.info(f"📊 Загружено посетителей из базы: {len(self.known_visitors_cache)}")

    def resize_frame_for_display(self, frame, target_width=1280):
        """Умное изменение размера кадра для отображения"""
        height, width = frame.shape[:2]

        # Если изображение уже меньше целевой ширины, не уменьшаем
        if width <= target_width:
            return frame

        # Рассчитываем новые размеры с сохранением пропорций
        ratio = target_width / width
        new_width = target_width
        new_height = int(height * ratio)

        # Ресайз с хорошей интерполяцией
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        return resized_frame

    def log_recognition_stats(self):
        """Логирование статистики распознавания"""
        current_time = time.time()
        if current_time - self.last_log_time >= 5.0:
            logger.info(f"📈 СТАТИСТИКА: Всего в базе: {len(self.known_visitors_cache)}, "
                        f"Активных треков: {len(self.face_tracks)}, "
                        f"Новых за сессию: {self.recognition_stats['new_visitors']}, "
                        f"Известных: {self.recognition_stats['known_visitors']}")
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
                                'is_confirmed': similarity > self.similarity_threshold
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
        """Быстрое получение эмбеддинга с улучшенной стабильностью"""
        try:
            face_resized = cv2.resize(face_image, (112, 112))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Нормализация яркости
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
        """Поиск лучшего совпадения с учетом временного кэша"""
        if embedding is None:
            return None, 0.0

        best_match_id = None
        best_similarity = 0.0

        for visitor_id, known_embedding in self.known_visitors_cache.items():
            similarity = self.calculate_similarity(embedding, known_embedding)
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
                    'coords': coords
                })
                face_data['track_id'] = best_track_id
                face_data['visitor_id'] = self.face_tracks[best_track_id].get('visitor_id')
                logger.debug(f"🔄 Обновлен трек {best_track_id}, схожесть: {best_similarity:.3f}")
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.face_tracks[track_id] = {
                    'embedding': embedding,
                    'last_seen': timestamp,
                    'coords': coords,
                    'visitor_id': None,
                    'created_at': timestamp
                }
                face_data['track_id'] = track_id
                logger.info(f"🎯 Создан новый трек {track_id}")

            updated_faces.append(face_data)

        return updated_faces

    def confirm_visitor_identity(self, track_id, face_data):
        """Подтверждение идентичности посетителя через несколько кадров"""
        track_data = self.face_tracks.get(track_id)
        if not track_data:
            return None

        embedding = face_data['embedding']
        timestamp = time.time()

        if track_data['visitor_id']:
            visitor_id = track_data['visitor_id']
            if visitor_id in self.known_visitors_cache:
                old_embedding = self.known_visitors_cache[visitor_id]
                new_embedding = 0.7 * old_embedding + 0.3 * embedding
                self.known_visitors_cache[visitor_id] = new_embedding / np.linalg.norm(new_embedding)

            self.recognition_stats['known_visitors'] += 1
            logger.debug(f"♻️  Подтвержден известный посетитель {visitor_id}")
            return visitor_id

        visitor_id, similarity = self.find_best_match(embedding)

        if similarity > self.similarity_threshold:
            track_data['visitor_id'] = visitor_id
            track_data['confirmed_at'] = timestamp
            self.recognition_stats['known_visitors'] += 1
            logger.info(f"👤 ОПОЗНАН известный посетитель {visitor_id}, схожесть: {similarity:.3f}")
            return visitor_id
        else:
            track_duration = timestamp - track_data['created_at']
            if track_duration > 2.0:
                new_visitor_id = self._create_new_visitor(embedding, track_id)
                if new_visitor_id:
                    self.recognition_stats['new_visitors'] += 1
                    logger.info(f"🆕 СОЗДАН новый посетитель {new_visitor_id}, схожесть с известными: {similarity:.3f}")
                return new_visitor_id
            else:
                logger.debug(f"⏳ Трек {track_id} ожидает подтверждения ({track_duration:.1f}s)")

        return None

    def _create_new_visitor(self, embedding, track_id):
        """Создание нового посетителя с подтверждением"""
        cursor = self.conn.cursor()
        now = datetime.datetime.now()

        embedding_blob = embedding.astype(np.float32).tobytes()
        cursor.execute(
            """INSERT INTO visitors (face_embedding, first_seen, last_seen, 
               visit_count, last_updated, confirmed_count) VALUES (?, ?, ?, 1, ?, 1)""",
            (embedding_blob, now, now, now)
        )
        visitor_id = cursor.lastrowid
        self.known_visitors_cache[visitor_id] = embedding
        self.conn.commit()

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
        logger.debug(f"📝 Обновлен визит посетителя {visitor_id}")

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
        """Применяет результаты обработки с трекингом"""
        processed_frame = frame.copy()
        processed_count = 0

        tracked_faces = self.update_face_tracking(result['faces'], current_time)

        for face_data in tracked_faces:
            visitor_id = self.confirm_visitor_identity(face_data['track_id'], face_data)

            if visitor_id:
                self.save_visitor_visit(visitor_id, face_data['embedding'])
                x, y, w, h = face_data['coords']

                is_new = face_data['similarity'] <= self.similarity_threshold
                color = (0, 0, 255) if is_new else (0, 255, 0)
                status = "NEW" if is_new else "KNOWN"

                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(processed_frame, f'{status}:{visitor_id}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(processed_frame, f'Sim:{face_data["similarity"]:.2f}',
                            (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                processed_count += 1

        self.log_recognition_stats()

        return processed_frame, result['detected_count'], processed_count

    def start_analysis(self, rtsp_url):
        """Запуск анализа с нормальным размером GUI"""
        logger.info("🚀 Запуск версии с нормальным GUI...")

        cap = self.setup_rtsp_camera(rtsp_url)
        if not cap.isOpened():
            return

        self.start_processing_thread()

        logger.info("✅ Анализ запущен")

        # Создаем окно нормального размера
        window_name = 'Trassir Visitor Analytics - NORMAL SIZE'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Устанавливаем начальный размер окна
        cv2.resizeWindow(window_name, 1280, 720)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("📡 Потеряно соединение с камерой...")
                    time.sleep(2)
                    continue

                processed_frame, detected, processed = self.process_frame_realtime(frame)

                # Умное масштабирование только если изображение слишком большое
                display_frame = self.resize_frame_for_display(processed_frame, target_width=1280)

                # Статистика на экране с читаемыми размерами
                stats_text = [
                    f"TRASSIR VISITOR ANALYTICS",
                    f"Detected: {detected}",
                    f"Processed: {processed}",
                    f"Total in DB: {len(self.known_visitors_cache)}",
                    f"Active Tracks: {len(self.face_tracks)}",
                    f"FPS: {self.current_fps:.1f}",
                    f"Press 'q' to quit"
                ]

                # Фон для текста для лучшей читаемости
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (450, 180), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

                for i, text in enumerate(stats_text):
                    cv2.putText(display_frame, text, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(window_name, display_frame)

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
            logger.info(f"   Известных обработано: {self.recognition_stats['known_visitors']}")
            logger.info("✅ Анализ завершен")

    def cleanup_database(self):
        """Очистка базы данных"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM visitors")
        self.conn.commit()
        self.known_visitors_cache.clear()
        self.face_tracks.clear()
        logger.info("🗑️ База данных очищена")


def main():
    """Основная функция"""
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    # Простые настройки
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