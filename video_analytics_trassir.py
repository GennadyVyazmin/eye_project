# video_analytics_trassir_optimized_fixed.py
import cv2
import numpy as np
import sqlite3
import datetime
import time
from deepface import DeepFace
import logging
import threading
from queue import Queue

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedTrassirCounter:
    def __init__(self, processing_interval=1.0, similarity_threshold=0.65):
        """
        Оптимизированная версия для снижения нагрузки на ЦП
        """
        self.conn = sqlite3.connect('visitors_trassir_opt.db', check_same_thread=False)
        self._init_database()

        self.processing_interval = processing_interval
        self.similarity_threshold = similarity_threshold

        # Трекинг состояния
        self.last_processing_time = 0
        self.known_visitors_cache = {}
        self.frame_count = 0
        self.last_frame = None
        self.processing_active = False

        # Очередь для асинхронной обработки
        self.frame_queue = Queue(maxsize=2)
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

        logger.info(f"Оптимизированная инициализация завершена. Интервал: {processing_interval}с")

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
                last_updated TIMESTAMP
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

        logger.info(f"Загружено посетителей: {len(self.known_visitors_cache)}")

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
                # Берем кадр из очереди с таймаутом
                frame_data = self.frame_queue.get(timeout=1.0)
                frame, frame_time = frame_data

                # Обработка кадра
                result = self._process_frame_heavy(frame)
                self.results_queue.put((result, frame_time))

                self.frame_queue.task_done()

            except:
                continue

    def _process_frame_heavy(self, frame):
        """Тяжелые операции обработки (выполняются в фоне)"""
        try:
            # Сильно уменьшаем разрешение для обработки
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame_small = cv2.resize(frame, (new_width, new_height))
            else:
                frame_small = frame

            # Детекция лиц на уменьшенном кадре
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            processed_faces = []
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Масштабируем координаты обратно
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
                            processed_faces.append({
                                'coords': (x_orig, y_orig, w_orig, h_orig),
                                'embedding': embedding
                            })

            return {
                'faces': processed_faces,
                'processed_count': len(processed_faces),
                'detected_count': len(faces)
            }

        except Exception as e:
            logger.error(f"Ошибка в фоновой обработке: {e}")
            return {'faces': [], 'processed_count': 0, 'detected_count': 0}

    def get_fast_embedding(self, face_image):
        """Быстрое получение эмбеддинга с оптимизацией"""
        try:
            # Сильное уменьшение для скорости
            face_resized = cv2.resize(face_image, (96, 96))

            # Минимальная предобработка
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            result = DeepFace.represent(
                face_rgb,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='skip',
                align=False
            )

            embedding = np.array(result[0]['embedding'], dtype=np.float32)

            # Проверяем что эмбеддинг не нулевой
            if np.all(embedding == 0):
                return None

            return embedding

        except Exception as e:
            logger.debug(f"Ошибка получения эмбеддинга: {e}")
            return None

    def calculate_similarity(self, embedding1, embedding2):
        """Безопасный расчет схожести с проверкой нулевых векторов"""
        if embedding1 is None or embedding2 is None:
            return 0.0

        try:
            # Проверяем на нулевые векторы
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Нормализация и расчет схожести
            emb1_norm = embedding1 / norm1
            emb2_norm = embedding2 / norm2

            similarity = float(np.dot(emb1_norm, emb2_norm))

            # Ограничиваем значение между 0 и 1
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

    def save_visitor(self, embedding):
        """Сохранение посетителя"""
        if embedding is None:
            return None

        cursor = self.conn.cursor()
        now = datetime.datetime.now()

        visitor_id, similarity = self.find_best_match(embedding)

        if similarity > self.similarity_threshold and visitor_id is not None:
            # Обновление существующего
            cursor.execute(
                "UPDATE visitors SET last_seen = ?, visit_count = visit_count + 1, last_updated = ? WHERE id = ?",
                (now, now, visitor_id)
            )
            logger.debug(f"Обновлен посетитель {visitor_id}, схожесть: {similarity:.3f}")
        else:
            # Новый посетитель
            embedding_blob = embedding.astype(np.float32).tobytes()
            cursor.execute(
                "INSERT INTO visitors (face_embedding, first_seen, last_seen, visit_count, last_updated) VALUES (?, ?, ?, 1, ?)",
                (embedding_blob, now, now, now)
            )
            visitor_id = cursor.lastrowid
            self.known_visitors_cache[visitor_id] = embedding
            logger.info(f"🆕 НОВЫЙ посетитель {visitor_id}, схожесть: {similarity:.3f}")

        self.conn.commit()
        return visitor_id

    def setup_rtsp_camera(self, rtsp_url):
        """Настройка RTSP с оптимизацией"""
        logger.info(f"Подключение к камере: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)

        # ОПТИМИЗАЦИЯ: Уменьшаем качество потока для предпросмотра
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

        # Пропускаем первые несколько кадров для стабилизации
        for _ in range(5):
            cap.read()

        # Проверяем подключение
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                logger.info(f"Камера подключена. Разрешение: {test_frame.shape[1]}x{test_frame.shape[0]}")
            else:
                logger.warning("Камера подключена, но не передает данные")
        else:
            logger.error("Не удалось подключиться к камере")

        return cap

    def process_frame_realtime(self, frame):
        """Обработка кадра в реальном времени (только отрисовка)"""
        current_time = time.time()

        # Обновляем FPS
        self.fps_frame_count += 1
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_frame_count / (current_time - self.fps_start_time)
            self.fps_frame_count = 0
            self.fps_start_time = current_time

        # Обрабатываем только каждый N-ый кадр
        if current_time - self.last_processing_time < self.processing_interval:
            # Но проверяем есть ли результаты от фоновой обработки
            try:
                result, frame_time = self.results_queue.get_nowait()
                return self._apply_processing_result(frame, result)
            except:
                return frame, 0, 0

        # Отправляем кадр в фоновую обработку
        if self.frame_queue.qsize() < 2:
            self.frame_queue.put((frame.copy(), current_time))

        self.last_processing_time = current_time

        # Пробуем получить результаты
        try:
            result, frame_time = self.results_queue.get_nowait()
            return self._apply_processing_result(frame, result)
        except:
            return frame, 0, 0

    def _apply_processing_result(self, frame, result):
        """Применяет результаты обработки к кадру"""
        processed_frame = frame.copy()
        processed_count = 0

        for face_data in result['faces']:
            x, y, w, h = face_data['coords']
            embedding = face_data['embedding']

            visitor_id = self.save_visitor(embedding)
            if visitor_id is not None:
                best_match_id, similarity = self.find_best_match(embedding)
                is_new = similarity <= self.similarity_threshold or best_match_id is None

                color = (0, 0, 255) if is_new else (0, 255, 0)
                status = "NEW" if is_new else "KNOWN"

                # Легкая отрисовка
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(processed_frame, f'{status}:{visitor_id}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                processed_count += 1

        return processed_frame, result['detected_count'], processed_count

    def start_analysis(self, rtsp_url):
        """Запуск оптимизированного анализа"""
        logger.info("Запуск оптимизированной версии...")

        cap = self.setup_rtsp_camera(rtsp_url)
        if not cap.isOpened():
            logger.error("Не удалось подключиться к камере")
            return

        # Запускаем фоновую обработку
        self.start_processing_thread()

        logger.info("🚀 Оптимизированный анализ запущен")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Потеряно соединение с камерой...")
                    time.sleep(2)
                    continue

                # Обработка кадра (только отрисовка)
                processed_frame, detected, processed = self.process_frame_realtime(frame)

                # Легкая статистика
                stats_text = [
                    f"TRASSIR OPTIMIZED",
                    f"Detected: {detected}",
                    f"Processed: {processed}",
                    f"Total: {len(self.known_visitors_cache)}",
                    f"FPS: {self.current_fps:.1f}",
                    f"Queue: {self.frame_queue.qsize()}",
                    f"Press 'q' to quit"
                ]

                # Простая отрисовка статистики
                for i, text in enumerate(stats_text):
                    cv2.putText(processed_frame, text, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Trassir - OPTIMIZED (Smooth Preview)', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Остановка по Ctrl+C...")
        except Exception as e:
            logger.error(f"Ошибка в основном цикле: {e}")
        finally:
            self.stop_processing = True
            if self.processing_thread:
                self.processing_thread.join(timeout=2.0)
            cap.release()
            cv2.destroyAllWindows()
            self.conn.close()
            logger.info("Анализ завершен")

    def cleanup_database(self):
        """Очистка базы данных"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM visitors")
        self.conn.commit()
        self.known_visitors_cache.clear()
        logger.info("База данных очищена")


def main():
    """Основная функция"""
    RTSP_URL = "rtsp://admin:admin@10.0.0.242:554/live/main"

    # Оптимизированные настройки
    counter = OptimizedTrassirCounter(
        processing_interval=1.0,
        similarity_threshold=0.65
    )

    # Раскомментируйте для очистки базы:
    # counter.cleanup_database()

    try:
        counter.start_analysis(RTSP_URL)
    except Exception as e:
        logger.error(f"Ошибка: {e}")


if __name__ == "__main__":
    main()