import csv
import os


class CSVLogger:
    """Minimal per-epoch scalar logger -- no TensorBoard dependency needed for
    a CPU-only, 4-week project; matplotlib turns these CSVs into paper figures."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fieldnames = None
        self._file = None
        self._writer = None

    def log(self, row: dict) -> None:
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
