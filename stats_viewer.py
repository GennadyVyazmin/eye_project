# stats_viewer.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def show_statistics():
    conn = sqlite3.connect('visitors.db')

    # Статистика за последние 24 часа
    query = """
    SELECT hour, unique_visitors, total_detections 
    FROM hourly_stats 
    WHERE hour >= datetime('now', '-24 hours')
    ORDER BY hour
    """

    df = pd.read_sql_query(query, conn)

    if not df.empty:
        df['hour'] = pd.to_datetime(df['hour'])

        plt.figure(figsize=(12, 6))
        plt.plot(df['hour'], df['unique_visitors'], marker='o', label='Уникальные посетители')
        plt.plot(df['hour'], df['total_detections'], marker='s', label='Всего детекций')
        plt.xlabel('Время')
        plt.ylabel('Количество')
        plt.title('Статистика посетителей по часам')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Общая статистика
    total_visitors = pd.read_sql_query("SELECT COUNT(*) as total FROM visitors", conn)
    print(f"Всего уникальных посетителей: {total_visitors['total'][0]}")

    conn.close()


if __name__ == "__main__":
    show_statistics()