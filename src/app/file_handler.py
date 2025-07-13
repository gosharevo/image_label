import json
import os
import random
import shutil
from pathlib import Path
from typing import List
from loguru import logger
from src.utils.config import SUPPORTED_IMAGE_FORMATS


def find_image_files(root_dir: str) -> List[str]:
    """
    Рекурсивно ищет все файлы изображений в указанной директории.
    Возвращает отсортированный список абсолютных путей в виде строк.
    """
    if not root_dir or not os.path.isdir(root_dir):
        return []

    if os.path.exists('filelist.json'):
        with open('filelist.json', 'r') as f:
            return json.load(f)

    logger.info(f"Начинаю рекурсивный поиск изображений в '{root_dir}'...")
    image_paths = []
    root_path = Path(root_dir)
    for ext in SUPPORTED_IMAGE_FORMATS:
        image_paths.extend(root_path.rglob(f"*{ext}"))

    # Конвертируем Path объекты в строки и сортируем для консистентности
    filtered_paths = [p for p in image_paths]
    filtered_paths.sort(key=lambda x: (str(x.parent.name) + str(x.name)).lower())
    sorted_paths = [str(p) for p in filtered_paths]
    logger.success(f"Найдено {len(sorted_paths)} изображений.")
    with open('filelist.json', 'w') as f:
        json.dump(sorted_paths, f)
    return sorted_paths


def generate_new_filename(image_path_str: str) -> str:
    """
    Генерирует новое имя файла по схеме: имя_родительской_папки_имя_файла.расширение.
    Учитывает +1 уровень вложенности для уменьшения коллизий.
    Пример: C:\\Users\\user\\images\\cats\\fluffy.jpg -> images_cats_fluffy.jpg
    """
    path = Path(image_path_str)
    parent = path.parent.name
    grandparent = path.parent.parent.name

    # Чтобы избежать коллизий между /a/b/img.jpg и /c/b/img.jpg, включаем grandparent
    # Если grandparent -- это корень диска, его имя будет пустым, что не страшно.
    if grandparent and not Path(image_path_str).parent.parent.is_absolute():
        new_name = f"{grandparent}_{parent}_{path.name}"
    else:
        new_name = f"{parent}_{path.name}"

    # Проверка на потенциально опасные символы для имен файлов
    new_name = "".join(c for c in new_name if c.isalnum() or c in ('_', '.', '-')).rstrip()
    return new_name


def safe_copy_file(src_path: str, dst_path: str):
    """
    Атомарное копирование файла. Сначала копирует во временный файл,
    затем переименовывает. Гарантирует, что на месте назначения не останется
    недокопированного файла при сбое.
    """
    try:
        dst_path_obj = Path(dst_path)
        dst_path_obj.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = dst_path_obj.with_suffix(dst_path_obj.suffix + ".tmp")
        shutil.copy2(src_path, tmp_path)  # copy2 сохраняет метаданные
        os.rename(tmp_path, dst_path_obj)
        logger.debug(f"Файл успешно скопирован из '{src_path}' в '{dst_path}'.")
    except Exception as e:
        logger.error(f"Ошибка при копировании файла из '{src_path}' в '{dst_path}': {e}")

        # Попытка удалить временный файл, если он остался
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise  # Пробрасываем исключение выше для отката транзакции


def safe_move_file(src_path: str, dst_path: str):
    """
    Безопасное перемещение файла. Создает целевую директорию, если ее нет.
    shutil.move достаточно умен для атомарного переименования на одной файловой системе.
    """
    try:
        dst_path_obj = Path(dst_path)
        dst_path_obj.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src_path, dst_path)
        logger.debug(f"Файл успешно перемещен из '{src_path}' в '{dst_path}'.")
    except Exception as e:
        logger.error(f"Ошибка при перемещении файла из '{src_path}' в '{dst_path}': {e}")
        raise


def safe_delete_file(file_path: str):
    """
    Безопасное удаление файла. Просто обертка для логирования и обработки ошибок.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Файл '{file_path}' успешно удален.")
    except Exception as e:
        logger.error(f"Ошибка при удалении файла '{file_path}': {e}")
        raise
