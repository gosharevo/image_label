# src/ml/file_handler.py

import os
from pathlib import Path
from typing import Generator
from loguru import logger


def find_image_files_recursively(folder_path: str) -> Generator[str, None, None]:
    """
    Рекурсивно находит все файлы изображений в указанной папке и отдает их по одному.

    Args:
        folder_path (str): Путь к папке для сканирования.

    Yields:
        Generator[str, None, None]: Пути к найденным файлам изображений.
    """
    folder = Path(folder_path)
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    logger.info(f"Начинается рекурсивный поиск изображений в: {folder}...")

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(supported_formats):
                yield str(Path(root) / file)

# Функции generate_new_filename, safe_copy_file, safe_move_file больше не нужны и удалены.

