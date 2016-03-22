from logging import getLogger, StreamHandler, DEBUG, INFO


def create_logger(name, debug=False):
    logger = None
    logger = getLogger(name)
    handler = StreamHandler()
    level = INFO if not debug else DEBUG
    handler.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
