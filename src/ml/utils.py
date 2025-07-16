# src/ml/utils.py

import torch
from torch import nn
from pathlib import Path
from typing import Tuple, Dict
from dataclasses import dataclass, asdict

from .model import ViTOrdinal
# Важно: импортируем базовые константы, чтобы иметь значения по умолчанию
from ..utils.config import (NUM_CLASSES, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS,
                            VAL_SPLIT_SIZE, RANDOM_STATE, DROP_PATH_RATE, WARMUP_EPOCHS,
                            EARLY_STOPPING_PATIENCE)


@dataclass
class TrainingConfig:
    """
    Централизованная и исчерпывающая конфигурация для процесса обучения.
    Этот дата-класс теперь является общей утилитой.
    """
    # Основные параметры
    num_classes: int = NUM_CLASSES
    learning_rate: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    num_epochs: int = NUM_EPOCHS
    val_split_size: float = VAL_SPLIT_SIZE
    random_state: int = RANDOM_STATE
    drop_path_rate: float = DROP_PATH_RATE
    warmup_epochs: int = WARMUP_EPOCHS

    # Параметры регуляризации и оптимизации
    weight_decay: float = 0.05
    min_lr: float = 1e-6
    gradient_clip_val: float = 1.0
    label_smoothing: float = 0.1
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE

    # ... можно добавить и остальные параметры из вашей последней версии тренера ...

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TrainingConfig':
        # Создаем экземпляр с параметрами из словаря, игнорируя лишние
        # ключи, если они есть в словаре, но нет в дата-классе.
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


def load_model_for_inference(
        model_path: Path,
        device: torch.device
) -> Tuple[nn.Module, TrainingConfig, torch.Tensor]:
    """
    Загружает модель и связанные с ней артефакты (конфигурацию, пороги) из чекпоинта.
    Это основная утилитарная функция для инициализации модели для инференса.

    Args:
        model_path (Path): Путь к файлу чекпоинта (.pth).
        device (torch.device): Устройство, на которое нужно загрузить модель ('cpu' или 'cuda').

    Returns:
        Tuple[nn.Module, TrainingConfig, torch.Tensor]: Кортеж, содержащий:
            - Загруженную и готовую к работе модель.
            - Экземпляр конфигурации, с которой модель обучалась.
            - Тензор с порогами для порядковой классификации.

    Raises:
        FileNotFoundError: Если файл модели не найден.
        KeyError: Если чекпоинт имеет неверный формат.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Файл модели не найден по пути: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Проверяем наличие ключевых полей в чекпоинте
    if 'config' not in checkpoint or 'model_state_dict' not in checkpoint:
        raise KeyError("Чекпоинт имеет неверный формат. Отсутствуют 'config' или 'model_state_dict'.")

    # 1. Восстанавливаем конфигурацию, с которой обучалась модель
    config = TrainingConfig.from_dict(checkpoint['config'])

    # 2. Создаем модель с той же архитектурой
    model = ViTOrdinal(
        num_classes=config.num_classes,
        pretrained=False,  # Веса будут загружены из файла, предзагрузка не нужна
        drop_path_rate=config.drop_path_rate
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Сразу переводим в режим инференса

    # 3. Загружаем оптимизированные пороги. Если их нет, используем стандартные 0.5
    default_thresholds = torch.full((config.num_classes - 1,), 0.5, device=device)
    thresholds = checkpoint.get('thresholds', default_thresholds)

    return model, config, thresholds