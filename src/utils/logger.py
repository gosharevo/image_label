 
import sys
from loguru import logger
from .config import LOG_FILE_PATH

def setup_logger():
    """
    Настраивает логгер loguru для вывода в консоль и в файл.
    Эта функция должна быть вызвана один раз при старте приложения.
    """
    logger.remove() # Удаляем стандартный обработчик

    # Логирование в консоль с уровнем INFO
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    # Логирование в файл с уровнем DEBUG для детальной диагностики
    logger.add(
        LOG_FILE_PATH,
        level="DEBUG",
        rotation="10 MB", # Ротация логов при достижении 10 MB
        retention="7 days", # Хранить логи за последние 7 дней
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    )

    logger.info("Логгер инициализирован.")