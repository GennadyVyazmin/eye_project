# create_database.py
import sqlite3
import datetime


def create_database():
    conn = sqlite3.connect('visitors.db')
    cursor = conn.cursor()

    # Таблица для хранения информации о посетителях
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_embedding BLOB,
            first_seen TIMESTAMP,
            last_seen TIMESTAMP,
            visit_count INTEGER DEFAULT 1
        )
    ''')

    # Таблица для почасовой статистики
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hourly_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour TIMESTAMP,
            unique_visitors INTEGER,
            total_detections INTEGER
        )
    ''')

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_database()