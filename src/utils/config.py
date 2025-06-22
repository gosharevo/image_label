import os
from pathlib import Path

# --- Базовые пути ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
LOGS_DIR = BASE_DIR / "logs"

# --- Файлы данных ---
DATA_JSON_PATH = BASE_DIR / "data.json"
PREDICTIONS_JSON_PATH = BASE_DIR / "predictions.json"
MODEL_PATH = MODEL_DIR / "latest.pt"
LOG_FILE_PATH = LOGS_DIR / "app.log"

# --- Параметры приложения ---
SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
INFERENCE_BATCH_SIZE = 10

# --- НОВАЯ НАСТРОЙКА: Стартовая папка для диалога выбора файлов ---
# Path.home() - это домашняя директория пользователя (e.g., C:/Users/YourUser)
# Можете заменить на конкретный путь: "F:/gosha/Iphone"
DEFAULT_START_FOLDER = str(Path(r"E:\Downloads\jpg"))
GRAD_CAM_DIR = BASE_DIR / "grad_cam_maps"

DEVICE = 'cuda'
# --- Параметры ML модели ---
IMG_SIZE = 224
NUM_CLASSES = 6
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
GRAD_MODE = "fade" # or jet

# --- Строки интерфейса ---
WINDOW_TITLE = "Ручная разметка изображений v1.1 (Pro Edition)"
MODEL_VERDICT_HEADER = "ВЕРДИКТ МОДЕЛИ:"
MODEL_PREDICTION_TEXT = "КЛАСС: {}\nВЕРОЯТНОСТЬ: {:.2f}"
MODEL_ALREADY_LABELED_TEXT = "[УЖЕ РАЗМЕЧЕНО]"
MODEL_NOT_TRAINED_TEXT = "[МОДЕЛЬ НЕ ОБУЧЕНА]"
MODEL_NO_PREDICTION_TEXT = "[НЕТ ПРЕДСКАЗАНИЯ]"