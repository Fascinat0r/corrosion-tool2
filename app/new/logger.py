import logging


def setup_logger(level=logging.INFO):
    logger = logging.getLogger("pipeline_logger")
    logger.setLevel(level)

    # Хэндлер для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Формат сообщений
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    # Добавим хэндлер к логгеру (избежим дублирования)
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger
