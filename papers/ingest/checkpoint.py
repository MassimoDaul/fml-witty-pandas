from pathlib import Path
from .config import DATA_DIR


def _path(offset: int) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"checkpoint_{offset}.txt"


def read_checkpoint(offset: int) -> int:
    """Return the last successfully processed line number, or offset if no checkpoint."""
    p = _path(offset)
    if p.exists():
        value = int(p.read_text().strip())
        print(f"Checkpoint found: resuming from line {value}")
        return value
    return offset


def write_checkpoint(offset: int, line_number: int) -> None:
    _path(offset).write_text(str(line_number))


def clear_checkpoint(offset: int) -> None:
    p = _path(offset)
    if p.exists():
        p.unlink()
        print(f"Checkpoint cleared for offset={offset}")
