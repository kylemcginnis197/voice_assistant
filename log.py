import logging

LOG_FILE = "session/session.log"

def setup_logging():
    """Call once at startup. Configures root logger to write to stdout and session.log."""
    fmt = logging.Formatter("[%(name)s] %(message)s")
    file_fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S")

    file_handler = logging.FileHandler(LOG_FILE, mode="w")  # 'w' replaces file each run
    file_handler.setFormatter(file_fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.WARNING)  # suppress third-party library noise
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    for name in ("main", "audio", "speech", "model", "tools"):
        logging.getLogger(name).setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
