# video_analytics_trassir_low_latency.py
import cv2
import numpy as np
import sqlite3
import datetime
import time
from deepface import DeepFace
import logging
import threading
from queue import Queue
import os

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


class LowLatencyCamera:
    def __init__(self, rtsp_url, target_fps=5):
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.cap = None
        self.last_frame = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        """Запуск потока чтения кадров"""
        logger.info(f"📡 Запуск камеры с низкой задержкой: {self.rtsp_url}")

        # Параметры для минимальной задержки
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

        if not self.cap.isOpened():
            logger.error("❌ Не удалось открыть камеру")
            return False

        # Очистка буфера камеры
        for _ in range(10):
            self.cap.read()

        self.running = True
        self.thread = threading.Thread(target=self._read_frames)
        self.thread.daemon = True
        self.thread.start()

        logger.info("✅ Поток чтения кадров запущен")
        return True

    def _read_frames(self):
        """Непрерывное чтение кадров в отдельном потоке"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.last_frame = frame.copy()
                        self.last_frame_time = time.time()
                        self.frame_count += 1
                else:
                    logger.warning("📡 Потеря кадра")
                    time.sleep(0.1)

                # Небольшая пауза для снижения нагрузки
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"❌ Ошибка чтения кадра: {e}")
                time.sleep(0.1)

    def get_frame(self):
        """Получение последнего кадра"""
        with self.lock:
            if self.last_frame is not None:
                return self.last_frame.copy(), self.last_frame_time
        return None, 0

    def stop(self):
        """Остановка камеры"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("🛑 Камера остановлена")


class LowLatencyCounter:
    def __init__(self, processing_interval=2.0, similarity_threshold=0.65):
        self.conn = sqlite3.connect('visitors_low_latency.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
        self.similarity_threshold = similarity_threshold

        # Настройки камеры
        self.target_fps = 5
        self.frame_interval = 1.0 / self.target_fps

        # Цвета для индикации статусов
        self.COLORS = {
            'detected': (0, 255, 0),  # Зеленый
            'tracking': (255, 255, 0),  # Желтый
            'known': (0, 255, 255),  # Голубой
            'new': (0, 0, 255),  # Красный
        }

        # Папки для хранения фото
        self.photos_dir = "visitor_photos_low_latency"
        self._create_directories()

        # Система трекинга
        self.face_tracks = {}
        self.next_track_id = 1
        self.track_max_age = 2.0

        # Статистика
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'faces_recognized': 0,
            'known_visitors': 0,
            'new_visitors': 0,
            'current_fps': 0.0,
            'camera_fps': 0.0,
            'processing_time': 0.0
        }

        # Детектор лиц
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Для расчета FPS
        self.fps_counter = 0
        self.fps_timer = time.time()

        # Кэш
        self.known_visitors_cache = {}
        self.embedding_cache = {}
        self._load_known_visitors()

        logger.info("🚀 Система с низкой задержкой инициализирована")

    def _create_directories(self):
        os.makedirs(self.photos_dir, exist_ok=True)

    def _init_database(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_embedding BLOB,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                visit_count INTEGER DEFAULT 1,
                photo_path TEXT
            )
        ''')
        self.conn.commit()

    def _load_known_visitors(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, face_embedding FROM visitors")
        for visitor_id, embedding_blob in cursor.fetchall():
            if embedding_blob:
                try:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    self.known_visitors_cache[visitor_id] = embedding
                except Exception as e:
                    logger.warning(f"Ошибка загрузки посетителя {visitor_id}: {e}")
        logger.info(f"📊 Загружено посетителей: {len(self.known_visitors_cache)}")

    def setup_camera_ultra_low_latency(self, rtsp_url):
        """Настройка камеры с ультра-низкой задержкой"""
        camera = LowLatencyCamera(rtsp_url, self.target_fps)
        if camera.start():
            # Ждем получения первых кадров
            for _ in range(20):  # 20 попыток
                frame, _ = camera.get_frame()
                if frame is not None:
                    logger.info(f"✅ Камера готова. Разрешение: {frame.shape[1]}x{frame.shape[0]}")
                    return camera
                time.sleep(0.1)
        return None

    def detect_faces_ultra_fast(self, frame):
        """Сверхбыстрая детекция лиц"""
        try:
            # Минимальный ресайз для скорости
            if frame.shape[1] > 800:
                small_frame = cv2.resize(frame, (800, int(800 * frame.shape[0] / frame.shape[1])))
            else:
                small_frame = frame

            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(80, 80),  # Увеличили для скорости
                maxSize=(400, 400),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Масштабируем координаты
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

            return scaled_faces

        except Exception as e:
            logger.error(f"❌ Ошибка детекции: {e}")
            return []

    def get_embedding_fast(self, face_image):
        """Быстрое получение эмбеддинга"""
        try:
            img_hash = hash(face_image.tobytes())
            if img_hash in self.embedding_cache:
                return self.embedding_cache[img_hash]

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

            # Кэширование
            if len(self.embedding_cache) > 50:
                self.embedding_cache.clear()
            self.embedding_cache[img_hash] = embedding

            return embedding

        except Exception as e:
            return None

    def update_face_tracking(self, faces, current_time):
        """Быстрое обновление трекинга"""
        active_tracks = {}

        # Очистка старых треков
        for track_id in list(self.face_tracks.keys()):
            if current_time - self.face_tracks[track_id]['last_seen'] > self.track_max_age:
                del self.face_tracks[track_id]

        # Обновление треков
        for face_bbox in faces:
            x, y, w, h = face_bbox
            face_center = (x + w // 2, y + h // 2)

            best_track_id = None
            best_distance = float('inf')

            for track_id, track_info in self.face_tracks.items():
                if current_time - track_info['last_seen'] > 1.0:
                    continue

                last_center = track_info['last_center']
                distance = np.sqrt((face_center[0] - last_center[0]) ** 2 +
                                   (face_center[1] - last_center[1]) ** 2)

                max_distance = min(w, h) * 1.5

                if distance < best_distance and distance < max_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None:
                self.face_tracks[best_track_id].update({
                    'last_seen': current_time,
                    'last_center': face_center,
                    'bbox': face_bbox,
                    'confirmed_count': self.face_tracks[best_track_id].get('confirmed_count', 0) + 1
                })
                active_tracks[best_track_id] = self.face_tracks[best_track_id]
            else:
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

    def process_frame_low_latency(self, frame):
        """Обработка кадра с минимальной задержкой"""
        start_time = time.time()
        current_time = start_time

        # Детекция лиц
        faces = self.detect_faces_ultra_fast(frame)

        # Трекинг
        active_tracks = self.update_face_tracking(faces, current_time)

        # Отрисовка
        processed_frame = frame.copy()

        for track_id, track_info in active_tracks.items():
            x, y, w, h = track_info['bbox']
            status = track_info.get('status', 'detected')
            color = self.COLORS[status]

            # Отрисовка bounding box
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)

            # Текст
            label = f"ID:{track_id} {status}"
            if track_info.get('visitor_id'):
                label += f" V:{track_info['visitor_id']}"

            cv2.putText(processed_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Обновление статистики
        processing_time = time.time() - start_time
        self.stats['processing_time'] = processing_time
        self.stats['frames_processed'] += 1
        self.stats['faces_detected'] += len(active_tracks)

        return processed_frame, len(active_tracks)

    def start_analysis_low_latency(self, rtsp_url):
        """Запуск анализа с ультра-низкой задержкой"""
        logger.info("🚀 Запуск режима УЛЬТРА-НИЗКОЙ ЗАДЕРЖКИ")

        # Настройка камеры
        camera = self.setup_camera_ultra_low_latency(rtsp_url)
        if not camera:
            logger.error("❌ Не удалось инициализировать камеру")
            return

        # Создание окна
        window_name = 'Trassir - ULTRA LOW LATENCY'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 900)

        logger.info("✅ Система готова. Задержка должна быть < 500ms")

        last_processing_time = 0
        frame_counter = 0

        try:
            while True:
                current_time = time.time()

                # Контроль FPS - обрабатываем каждые 200ms
                if current_time - last_processing_time < self.frame_interval:
                    time.sleep(0.001)  # Короткая пауза
                    continue

                last_processing_time = current_time

                # Получение кадра от камеры
                frame, frame_time = camera.get_frame()
                if frame is None:
                    continue

                # Расчет задержки кадра
                frame_delay = current_time - frame_time

                # Обработка кадра
                processed_frame, faces_count = self.process_frame_low_latency(frame)

                # Обновление FPS
                frame_counter += 1
                if current_time - self.fps_timer >= 1.0:
                    self.stats['current_fps'] = frame_counter / (current_time - self.fps_timer)
                    self.stats['camera_fps'] = camera.frame_count / (current_time - self.fps_timer)
                    camera.frame_count = 0
                    frame_counter = 0
                    self.fps_timer = current_time

                # Подготовка к отображению
                display_frame = processed_frame
                if display_frame.shape[1] > 1200:
                    display_frame = cv2.resize(display_frame,
                                               (1200, int(1200 * display_frame.shape[0] / display_frame.shape[1])))

                # Статистика на экране
                stats_text = [
                    f"ULTRA LOW LATENCY MODE",
                    f"Camera FPS: {self.stats['camera_fps']:.1f} | UI FPS: {self.stats['current_fps']:.1f}",
                    f"Frame delay: {frame_delay * 1000:.0f}ms | Process: {self.stats['processing_time'] * 1000:.1f}ms",
                    f"Active faces: {faces_count} | Tracks: {len(self.face_tracks)}",
                    f"Known: {self.stats['known_visitors']} | New: {self.stats['new_visitors']}",
                    f"Press Q to quit"
                ]

                # Отрисовка статистики
                for i, text in enumerate(stats_text):
                    y_pos = 30 + i * 25
                    cv2.rectangle(display_frame, (5, y_pos - 20), (550, y_pos + 5), (0, 0, 0), -1)
                    cv2.putText(display_frame, text, (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Индикатор задержки
                delay_color = (0, 255, 0) if frame_delay < 0.5 else (0, 255, 255) if frame_delay < 1.0 else (0, 0, 255)
                cv2.putText(display_frame, f"LATENCY: {frame_delay * 1000:.0f}ms",
                            (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, delay_color, 2)

                cv2.imshow(window_name, display_frame)

                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("⏹️ Остановка...")
        finally:
            camera.stop()
            cv2.destroyAllWindows()
            self.conn.close()

            logger.info(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
            logger.info(f"   Средний FPS: {self.stats['current_fps']:.1f}")
            logger.info(f"   Задержка обработки: {self.stats['processing_time'] * 1000:.1f}ms")
            logger.info(f"   Обнаружено лиц: {self.stats['faces_detected']}")
            logger.info("✅ Анализ завершен")


def main():
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    counter = LowLatencyCounter(
        processing_interval=2.0,
        similarity_threshold=0.65
    )

    try:
        counter.start_analysis_low_latency(RTSP_URL)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()