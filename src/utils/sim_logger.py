from __future__ import annotations

from pathlib import Path


class SimLogger:
    def __init__(self) -> None:
        self._file = None

    def configure(self, log_path: Path) -> None:
        if self._file is not None:
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", buffering=1)

    def log(self, msg: str) -> None:
        print(msg)
        if self._file:
            self._file.write(msg + "\n")

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


logger = SimLogger()
