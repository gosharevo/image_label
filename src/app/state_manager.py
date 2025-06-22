import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger

from . import file_handler
from ..utils.config import DATA_JSON_PATH, PREDICTIONS_JSON_PATH, DATA_DIR


class StateManager:
    """
    Управляет состоянием приложения: списки файлов, аннотации, предсказания.
    Обеспечивает атомарность операций с JSON и файлами.
    """

    def __init__(self):
        self.all_image_paths: List[str] = []
        self.annotations: Dict[str, int] = {}
        self.predictions: Dict[str, List] = {}
        self._load_state()

    def _load_state(self):
        """Загружает аннотации и предсказания из JSON файлов, устойчиво к ошибкам."""
        for path, state_dict_name in [(DATA_JSON_PATH, "annotations"), (PREDICTIONS_JSON_PATH, "predictions")]:
            try:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        setattr(self, state_dict_name, data)
                    logger.info(f"Загружено {len(getattr(self, state_dict_name))} записей из {path}")
                else:
                    logger.warning(
                        f"Файл {path} не найден или пуст. Начинаем с пустым состоянием для '{state_dict_name}'.")
                    setattr(self, state_dict_name, {})
            except json.JSONDecodeError as e:
                logger.error(
                    f"Ошибка декодирования JSON в файле {path}. Файл может быть поврежден. Сбрасываю состояние. Ошибка: {e}")
                setattr(self, state_dict_name, {})  # Сброс до пустого словаря при ошибке
            except Exception as e:
                logger.error(f"Не удалось загрузить состояние из {path}: {e}")

    def _save_state_atomic(self, file_path: Path, data: Dict):
        """Атомарно сохраняет словарь в JSON файл."""
        temp_path = file_path.with_suffix('.json.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            # os.replace является атомарной операцией и перезаписывает файл, если он существует (кроссплатформенно)
            os.replace(temp_path, file_path)
        except Exception as e:
            logger.critical(f"Критическая ошибка при сохранении файла состояния {file_path}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Пробрасываем ошибку, чтобы транзакция могла быть отменена
            raise

    def load_images_from_folder(self, folder_path: str):
        self.all_image_paths = file_handler.find_image_files(folder_path)

        # Фильтрация только существующих аннотаций
        paths_set = set(self.all_image_paths)
        prev_len = len(self.annotations)
        self.annotations = {p: l for p, l in self.annotations.items() if p in paths_set}
        if len(self.annotations) != prev_len:
            logger.warning("Очищены аннотации для отсутствующих файлов.")
            # Немедленно сохраняем очищенное состояние
            try:
                self._save_state_atomic(DATA_JSON_PATH, self.annotations)
            except Exception:
                # Если сохранение не удалось, логируем, но не падаем
                logger.error("Не удалось сохранить очищенные аннотации.")

    def get_unannotated_images(self) -> List[str]:
        annotated_set = set(self.annotations.keys())
        return [p for p in self.all_image_paths if p not in annotated_set]

    def update_annotation(self, image_path: str, new_label: int) -> bool:
        """
        Главная транзакционная функция. Обновляет аннотацию, перемещает/копирует файл.
        В случае любой ошибки откатывает изменения.
        """
        old_label = self.annotations.get(image_path)
        new_filename = file_handler.generate_new_filename(image_path)
        new_dst_path = str(DATA_DIR / str(new_label) / new_filename)

        # Создаем копию аннотаций для отката
        original_annotations = self.annotations.copy()

        try:
            # Сначала обновляем состояние в памяти. Это наш "коммит".
            self.annotations[image_path] = new_label
            # Теперь пытаемся синхронизировать файловую систему и сохранить JSON.

            # Сохраняем JSON. Если это не удастся, вылетит исключение, файловые операции не начнутся.
            self._save_state_atomic(DATA_JSON_PATH, self.annotations)

            # Если JSON сохранен, выполняем файловые операции.
            if old_label is not None:
                if old_label != new_label:
                    old_filename = file_handler.generate_new_filename(image_path)
                    old_dst_path = str(DATA_DIR / str(old_label) / old_filename)
                    if os.path.exists(old_dst_path):
                        file_handler.safe_move_file(old_dst_path, new_dst_path)
                    else:
                        logger.warning(
                            f"Исходный файл {old_dst_path} не найден для перемещения. Только запись в JSON обновлена.")
            else:
                file_handler.safe_copy_file(image_path, new_dst_path)

            logger.info(f"Аннотация для '{image_path}' обновлена на класс {new_label}.")
            return True

        except Exception as e:
            # Откат изменений
            logger.error(f"Транзакция по обновлению аннотации для '{image_path}' провалена. Откат. Ошибка: {e}")
            self.annotations = original_annotations  # Восстанавливаем состояние в памяти
            # Пытаемся сохранить на диск откаченное состояние
            try:
                self._save_state_atomic(DATA_JSON_PATH, self.annotations)
                logger.info("Состояние аннотаций успешно отменено и сохранено.")
            except Exception as save_err:
                logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось даже откатить состояние data.json! Ошибка: {save_err}")
            return False

    def update_predictions(self, new_predictions: Dict[str, List]):
        self.predictions.update(new_predictions)
        try:
            self._save_state_atomic(PREDICTIONS_JSON_PATH, self.predictions)
            logger.info(f"Сохранено {len(new_predictions)} новых предсказаний.")
        except Exception as e:
            logger.error(f"Не удалось сохранить предсказания: {e}")

    def get_full_dataset_for_training(self) -> List[Tuple[str, int]]:
        dataset = []
        for img_path, label in self.annotations.items():
            filename = file_handler.generate_new_filename(img_path)
            dataset_filepath = str(DATA_DIR / str(label) / filename)
            if os.path.exists(dataset_filepath):
                dataset.append((dataset_filepath, label))
            else:
                logger.warning(f"Файл {dataset_filepath} для обучения не найден, пропущен.")
        return dataset