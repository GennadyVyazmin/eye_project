# video_analytics_trassir_deepsort.py
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

# Попробуем импортировать DeepSORT
try:
    from deep_sort_realtime import DeepSort

    DEEPSORT_AVAILABLE = True
    logger.info("✅ DeepSORT доступен")
except ImportError:
    DEEPSORT_AVAILABLE = False
    logger.warning("❌ DeepSORT не установлен. Используем базовый трекинг")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
                self.deepsort = DeepSort(
                    max_age=30,  # Максимальное время жизни трека без обновления
                    n_init=3,  # Количество кадров для инициализации трека
                    max_cosine_distance=0.4,  # Максимальное косинусное расстояние для ассоциации
                    nn_budget=100  # Бюджет нейросети для внешних признаков
                )
                logger.info("🚀 DeepSORT инициализирован для трекинга по внешности")
            except Exception as e:
                logger.error(f"❌ Ошибка инициализации DeepSORT: {e}")
                self.deepsort = None

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

        logger.info(f"Продвинутая инициализация с DeepSORT: {DEEPSORT_AVAILABLE}")

    def _create_directories(self):
        """Создание папок для хранения фото"""
        os.makedirs(self.photos_dir, exist_ok=True)
        os.makedirs(os.path.join(self.photos_dir, self.current_session_dir), exist_ok=True)

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

    def extract_appearance_features(self, full_body_image):
        """Извлечение признаков внешности (одежда, телосложение)"""
        try:
            if not DEEPSORT_AVAILABLE or self.deepsort is None:
                return None

            # DeepSORT автоматически извлекает признаки при детекции
            # Здесь мы можем добавить дополнительную обработку
            height, width = full_body_image.shape[:2]

            # Простые гистограммы цвета как дополнительные признаки
            hsv = cv2.cvtColor(full_body_image, cv2.COLOR_BGR2HSV)
            hist_hue = cv2.calcHist([hsv], [0], None, [8], [0, 180])
            hist_sat = cv2.calcHist([hsv], [1], None, [4], [0, 256])
            hist_val = cv2.calcHist([hsv], [2], None, [4], [0, 256])

            # Нормализация гистограмм
            hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()
            hist_sat = cv2.normalize(hist_sat, hist_sat).flatten()
            hist_val = cv2.normalize(hist_val, hist_val).flatten()

            # Объединяем все признаки
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
            # Косинусная схожесть для гистограмм
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(features1, features2) / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.debug(f"Ошибка расчета схожести внешности: {e}")
            return 0.0

    def process_with_deepsort(self, frame, faces):
        """Обработка кадра с DeepSORT"""
        if not DEEPSORT_AVAILABLE or self.deepsort is None:
            return []

        try:
            # Подготавливаем детекции для DeepSORT
            detections = []
            for (x, y, w, h) in faces:
                # Расширяем bounding box для захвата большего участка тела
                expansion = 0.3  # 30% расширение
                x_exp = max(0, int(x - w * expansion))
                y_exp = max(0, int(y - h * expansion))
                w_exp = min(frame.shape[1] - x_exp, int(w * (1 + 2 * expansion)))
                h_exp = min(frame.shape[0] - y_exp, int(h * (1 + 2 * expansion)))

                confidence = 0.9  # Высокая уверенность для лиц
                detections.append(([x_exp, y_exp, w_exp, h_exp], confidence, None))

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

                # Извлекаем признаки внешности
                appearance_features = track.features if hasattr(track, 'features') else None

                deepsort_results.append({
                    'track_id': track_id,
                    'coords': (x, y, w, h),
                    'appearance_features': appearance_features,
                    'type': 'deepsort'
                })

                # Обновляем кэш треков DeepSORT
                self.deepsort_tracks[track_id] = {
                    'coords': (x, y, w, h),
                    'appearance_features': appearance_features,
                    'last_seen': time.time()
                }

            return deepsort_results

        except Exception as e:
            logger.error(f"Ошибка DeepSORT: {e}")
            return []

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
            if normalized_distance > 0.7:  # Порог близости
                if normalized_distance > best_similarity:
                    best_similarity = normalized_distance
                    best_track_id = track_id

        return best_track_id, best_similarity

    def combined_similarity_score(self, face_similarity, appearance_similarity, spatial_similarity):
        """Комбинированная оценка схожести"""
        # Весовые коэффициенты
        face_weight = 0.6  # Лицо - самый важный признак
        appearance_weight = 0.3  # Внешность (одежда)
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

            # Пространственная схожесть (в данном контексте не применяется)
            spatial_similarity = 0.5  # Нейтральное значение

            # Комбинированная оценка
            combined_similarity = self.combined_similarity_score(
                face_similarity, appearance_similarity, spatial_similarity
            )

            if combined_similarity > best_combined_similarity:
                best_combined_similarity = combined_similarity
                best_match_id = visitor_id

        # Логируем тип совпадения
        if best_match_id:
            if best_combined_similarity > self.similarity_threshold:
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

    # ... (остальные методы save_visitor_photo, update_visitor_gallery, create_gallery_display и т.д.)
    # остаются аналогичными предыдущей версии, но с добавлением работы с appearance_features

    def _create_new_visitor(self, embedding, face_image, appearance_features, track_id):
        """Создание нового посетителя с признаками внешности"""
        cursor = self.conn.cursor()
        now = datetime.datetime.now()

        visitor_id = None
        try:
            # Сохраняем фото
            photo_path = self.save_visitor_photo(face_image, visitor_id)

            # Подготавливаем бинарные данные
            embedding_blob = embedding.astype(np.float32).tobytes()
            appearance_blob = appearance_features.astype(
                np.float32).tobytes() if appearance_features is not None else None

            cursor.execute(
                """INSERT INTO visitors (face_embedding, first_seen, last_seen, 
                   visit_count, last_updated, confirmed_count, photo_path, appearance_features) 
                   VALUES (?, ?, ?, 1, ?, 1, ?, ?)""",
                (embedding_blob, now, now, now, photo_path, appearance_blob)
            )
            visitor_id = cursor.lastrowid

            self.known_visitors_cache[visitor_id] = {
                'embedding': embedding,
                'photo_path': photo_path,
                'appearance_features': appearance_features
            }
            self.conn.commit()

        except Exception as e:
            logger.error(f"❌ Ошибка создания посетителя: {e}")
            self.conn.rollback()
            return None

        if track_id in self.face_tracks:
            self.face_tracks[track_id]['visitor_id'] = visitor_id

        return visitor_id

    def start_analysis(self, rtsp_url):
        """Запуск анализа с DeepSORT"""
        logger.info("🚀 Запуск продвинутой версии с DeepSORT...")

        if not DEEPSORT_AVAILABLE:
            logger.warning("⚠️  DeepSORT недоступен. Установите: pip install deep-sort-realtime")

        cap = self.setup_rtsp_camera(rtsp_url)
        if not cap.isOpened():
            return

        self.start_processing_thread()
        logger.info("✅ Продвинутый анализ запущен")

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

                # Статистика с информацией о DeepSORT
                stats_text = [
                    f"ADVANCED ANALYTICS WITH DEEPSORT",
                    f"Detected: {detected}",
                    f"Processed: {processed}",
                    f"DeepSORT: {'ON' if DEEPSORT_AVAILABLE else 'OFF'}",
                    f"DeepSORT matches: {self.recognition_stats['deepsort_matches']}",
                    f"Face matches: {self.recognition_stats['face_only_matches']}",
                    f"FPS: {self.current_fps:.1f}",
                    f"Press 'q' to quit"
                ]

                overlay = display_with_gallery.copy()
                cv2.rectangle(overlay, (0, 0), (550, 200), (0, 0, 0), -1)
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
            logger.info(f"   Совпадений по DeepSORT: {self.recognition_stats['deepsort_matches']}")
            logger.info(f"   Совпадений только по лицу: {self.recognition_stats['face_only_matches']}")
            logger.info("✅ Анализ завершен")


def main():
    """Основная функция"""
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    counter = AdvancedTrassirCounter(
        processing_interval=1.0,  # Уменьшили интервал для лучшего трекинга
        similarity_threshold=0.55,
        tracking_threshold=0.45
    )

    try:
        counter.start_analysis(RTSP_URL)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()