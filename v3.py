try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    try:
        from deep_sort_realtime import DeepSort
    except ImportError:
        # используем базовый трекинг