# src/utils/config.py

import torch
from pathlib import Path

# --- Базовая директория проекта ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- КЛЮЧЕВЫЕ ПУТИ К ДАННЫМ ---

# 1. Файл-база данных аннотаций (Единый источник правды)
# Этот JSON хранит отображение {оригинальный_путь: метка}.
# Модель теперь будет обучаться напрямую на путях из этого файла.
DATA_PATH = BASE_DIR / "data.json"
LOG_FILE_PATH = BASE_DIR / 'logs' / 'app.log'

# 2. Папка для резервных копий файла аннотаций
BACKUP_PATH = BASE_DIR / "backups"
BACKUP_PATH.mkdir(exist_ok=True)

# 3. Пути для моделей
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "best_model.pth"

# 4. Папка для кэша
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# --- Параметры модели и обучения ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DROP_PATH_RATE = 0.1

# --- Параметры разделения данных и планировщика ---
VAL_SPLIT_SIZE = 0.15
RANDOM_STATE = 42
WARMUP_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 10

# --- Параметры UI и инференса ---
WINDOW_TITLE = "Система разметки и анализа изображений v3.1 (Упрощенная)"
DEFAULT_START_FOLDER = r"E:\Downloads\jpg"
INFERENCE_BATCH_SIZE = 5

# --- Текстовые константы для UI ---
MODEL_VERDICT_HEADER = "Вердикт модели"
MODEL_PREDICTION_TEXT = "Предсказание: <b>Класс {}</b><br>Уверенность: <b>{:.1%}</b>"
MODEL_ALREADY_LABELED_TEXT = "Изображение оценено: <b>Класс {}</b>"
MODEL_NOT_TRAINED_TEXT = "Модель не обучена."
MODEL_NO_PREDICTION_TEXT = "Предсказания для этого файла нет.\nНажмите 'Получить подсказки'."

