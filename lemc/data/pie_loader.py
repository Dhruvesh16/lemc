import sys

from .paths import BENCHMARK_REPO


def _ensure_benchmark_on_path():
    if BENCHMARK_REPO not in sys.path:
        sys.path.insert(0, BENCHMARK_REPO)


def load_pie_database(data_root: str) -> dict:
    _ensure_benchmark_on_path()
    from pie_data import PIE

    pie = PIE(data_path=data_root)
    return pie.generate_database()


def load_pie_splits(data_root: str) -> dict:
    _ensure_benchmark_on_path()
    from pie_data import PIE

    pie = PIE(data_path=data_root)
    return {
        "train": pie._get_image_set_ids("train"),
        "val": pie._get_image_set_ids("val"),
        "test": pie._get_image_set_ids("test"),
    }
