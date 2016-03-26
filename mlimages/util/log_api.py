import os
from datetime import datetime
import types
from logging import getLogger, DEBUG, INFO
from logging import StreamHandler, FileHandler
from mlimages.util.file_api import FileAPI


def _bool_2_level(boolean):
    level = INFO if not boolean else DEBUG
    return level


def __close(self):
    handlers = self.handlers[:]
    for h in handlers:
        h.close()
        self.removeHandler(h)


def create_logger(name, debug=False):
    logger = None
    logger = getLogger(name)
    handler = StreamHandler()
    level = _bool_2_level(debug)
    handler.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def create_file_logger(root, name, file_name="log.txt", timestamp_format="", debug=False):
    file_api = FileAPI(root)
    timestamp = ""
    if timestamp_format:
        timestamp = datetime.now().strftime(timestamp_format)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder = name + "_" + timestamp
    # prepare folder and file
    with file_api.open_with_mkdir(folder + "/" + file_name) as f:
        f.write("".encode("utf-8"))

    log_root = os.path.join(root, folder)
    logger = create_logger(name, debug)
    fh = FileHandler(os.path.join(log_root, file_name), encoding="utf-8")
    fh.setLevel(_bool_2_level(debug))
    logger.addHandler(fh)

    # add close method to release resource
    logger.close = types.MethodType(__close, logger)
    return logger, log_root
