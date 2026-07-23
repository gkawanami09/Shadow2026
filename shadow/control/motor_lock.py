"""Lock entre processos para garantir um unico dono dos motores/serial."""

import os
from pathlib import Path
import tempfile


DEFAULT_LOCK_PATH = Path(tempfile.gettempdir()) / "shadow2026-motors.lock"


class MotorLockError(RuntimeError):
    pass


class MotorOwnerLock:
    """Lock de sistema liberado automaticamente quando o processo encerra."""

    def __init__(self, owner, path=None):
        self.owner = str(owner)
        self.path = Path(path) if path is not None else DEFAULT_LOCK_PATH
        self._file = None
        self._locked = False

    def acquire(self):
        if self._locked:
            return self

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a+", encoding="utf-8")
        try:
            if os.name == "nt":
                self._acquire_windows()
            else:
                self._acquire_posix()
        except (OSError, BlockingIOError) as err:
            current_owner = self._read_owner()
            self._file.close()
            self._file = None
            suffix = (
                f" (dono registrado: {current_owner})"
                if current_owner else "")
            raise MotorLockError(
                "os motores ja estao reservados por outro processo"
                f"{suffix}") from err

        self._locked = True
        self._file.seek(0)
        self._file.truncate()
        self._file.write(f"pid={os.getpid()} owner={self.owner}\n")
        self._file.flush()
        return self

    def _acquire_posix(self):
        import fcntl
        fcntl.flock(
            self._file.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _acquire_windows(self):
        import msvcrt
        self._file.seek(0, os.SEEK_END)
        if self._file.tell() == 0:
            self._file.write("\0")
            self._file.flush()
        self._file.seek(0)
        msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)

    def _read_owner(self):
        try:
            self._file.seek(0)
            return self._file.read().strip().replace("\0", "")
        except (OSError, ValueError):
            return ""

    def release(self):
        if self._file is None:
            return
        try:
            if self._locked:
                if os.name == "nt":
                    import msvcrt
                    self._file.seek(0)
                    msvcrt.locking(
                        self._file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._locked = False
            self._file.close()
            self._file = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

