# src/app/state_manager.py

import json
import os
import shutil
from pathlib import Path
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple
from loguru import logger
from PyQt5.QtCore import QObject, pyqtSignal

# Импортируем из нашего обновленного модуля
from src.app import file_handler
# Импортируем константы из центрального конфига
from src.utils.config import DATA_PATH, BACKUP_PATH
from src.ml.predictor import PredictionResult


class AppState(Enum):
    IDLE = auto()
    TRAINING = auto()
    INFERRING = auto()
    LOADING = auto()


class StateManager(QObject):
    state_changed = pyqtSignal(AppState)
    annotations_changed = pyqtSignal()
    predictions_changed = pyqtSignal()
    file_list_updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._app_state: AppState = AppState.IDLE
        self._current_image_path: Optional[str] = None

        self.all_image_paths: List[str] = []
        self.annotations: Dict[str, int] = {}
        self.predictions: Dict[str, PredictionResult] = {}

        self._load_annotations()

        from src.utils.config import NUM_CLASSES
        self.config = lambda: None
        self.config.num_classes = NUM_CLASSES

    @property
    def app_state(self) -> AppState:
        return self._app_state

    def set_app_state(self, new_state: AppState):
        if self._app_state != new_state:
            logger.debug(f"Состояние приложения: {self._app_state.name} -> {new_state.name}")
            self._app_state = new_state
            self.state_changed.emit(new_state)

    @property
    def current_image_path(self) -> Optional[str]:
        return self._current_image_path

    def set_current_image(self, path: Optional[str]):
        if self._current_image_path != path:
            self._current_image_path = path
            self.state_changed.emit(self.app_state)

    def _load_annotations(self):
        try:
            if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
                with open(DATA_PATH, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
                logger.info(f"Загружено {len(self.annotations)} аннотаций из {DATA_PATH}")
            else:
                self.annotations = {}
        except Exception as e:
            logger.error(f"Ошибка загрузки аннотаций {DATA_PATH}: {e}. Начинаем с пустым словарем.")
            self.annotations = {}

    def _save_annotations_atomic(self):
        temp_path = DATA_PATH.with_suffix('.json.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, DATA_PATH)
            logger.debug("Файл аннотаций атомарно сохранен.")
        except Exception as e:
            logger.error(f"Критическая ошибка при сохранении {DATA_PATH}: {e}")
            if temp_path.exists(): os.remove(temp_path)
            # В этой упрощенной модели мы не будем делать откат, просто сообщим об ошибке.
            # Для надежности можно было бы пробросить исключение, но это усложнит UI.

    def load_images_from_folder(self, folder_path: str):
        self.set_app_state(AppState.LOADING)
        try:
            image_paths_from_scan = list(file_handler.find_image_files_recursively(folder_path))
            self.all_image_paths = sorted(image_paths_from_scan, key=lambda x: Path(x).parent.name.lower())
            logger.success(f"Завершено. Всего найдено {len(self.all_image_paths)} изображений.")
            self.file_list_updated.emit()
        except Exception as e:
            logger.error(f"Ошибка при сканировании папки: {e}")
        finally:
            self.set_app_state(AppState.IDLE)

    def update_annotation(self, image_path: str, new_label: int):
        """
        Упрощенная функция. Просто обновляет словарь и сохраняет JSON.
        Никаких операций с файлами.
        """
        try:
            # Шаг 1: Обновить состояние в памяти
            self.annotations[image_path] = new_label
            # Шаг 2: Сохранить на диск
            self._save_annotations_atomic()
            # Шаг 3: Оповестить UI
            self.annotations_changed.emit()
            logger.info(f"Аннотация для '{Path(image_path).name}' обновлена на класс {new_label}.")
        except Exception as e:
            logger.error(f"Не удалось сохранить аннотацию для {image_path}: {e}")
            # Откат не нужен, т.к. ошибка произошла при сохранении, в памяти останется новое значение,
            # но при следующем запуске загрузится старое, что является консистентным поведением.

    def update_predictions(self, new_predictions: Dict[str, PredictionResult]):
        serializable_preds = {path: result.__dict__ for path, result in new_predictions.items()}
        self.predictions.update(new_predictions)

        try:
            predictions_path = DATA_PATH.parent / "predictions.json"
            with open(predictions_path, 'w', encoding='utf-8') as f:
                for v in serializable_preds.values():
                    if isinstance(v['visualization_path'], Path): v['visualization_path'] = str(v['visualization_path'])
                json.dump(serializable_preds, f, indent=2)
        except Exception as e:
            logger.error(f"Не удалось сохранить предсказания: {e}")

        self.predictions_changed.emit()
        self.state_changed.emit(self.app_state)

    def get_full_dataset_for_training(self) -> List[Tuple[str, int]]:
        """
        Возвращает датасет для обучения. Теперь он состоит из оригинальных путей.
        Trainer будет работать с ними напрямую.
        """
        return list(self.annotations.items())

    def get_unannotated_images(self) -> List[str]:
        return [path for path in self.all_image_paths if path not in self.annotations]

    def backup_annotations(self):
        if not DATA_PATH.exists(): return
        try:
            from datetime import datetime
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            BACKUP_PATH.mkdir(parents=True, exist_ok=True)
            backup_file = BACKUP_PATH / f"data.{now_str}.json"
            shutil.copy(DATA_PATH, backup_file)
            logger.success(f"Бэкап аннотаций создан: {backup_file}")
        except Exception as e:
            logger.error(f"Ошибка при создании бэкапа: {e}")


