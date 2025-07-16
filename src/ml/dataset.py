# src/ml/dataset.py

from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Callable
from PIL import Image, UnidentifiedImageError
from loguru import logger
import torch

Image.MAX_IMAGE_PIXELS = None

class ImageLabelDataset(Dataset):
    """
    Датасет, который принимает список пар (путь_к_файлу, метка).
    Ключевое улучшение: он теперь отказоустойчив. Если изображение
    не удается открыть, он логирует ошибку и возвращает None,
    чтобы DataLoader мог его отфильтровать.
    """

    def __init__(self, file_list: List[Tuple[str, int]], transform: Optional[Callable] = None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, int]]:
        """
        Загружает одно изображение и его метку.
        Возвращает None в случае любой ошибки чтения файла.
        """
        img_path, label = self.file_list[index]

        try:
            # Открываем изображение
            image = Image.open(img_path).convert("RGB")

            # Применяем трансформации, если они есть
            if self.transform:
                image_tensor = self.transform(image)

            return image_tensor, label

        except FileNotFoundError:
            logger.warning(f"Файл не найден, пропущен: {img_path}")
            return None
        except (IOError, UnidentifiedImageError, ValueError) as e:
            logger.warning(f"Файл поврежден или не может быть открыт, пропущен: {img_path}. Ошибка: {e}")
            return None
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при обработке файла {img_path}: {e}")
            return None


class InferenceDataset(Dataset):
    """
    Отказоустойчивый датасет для инференса (возвращает путь к файлу).
    """

    def __init__(self, file_paths: List[str], transform: Optional[Callable] = None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, str]]:
        img_path = self.file_paths[index]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image_tensor = self.transform(image)
            return image_tensor, img_path
        except Exception as e:
            logger.warning(f"Файл для инференса не может быть открыт, пропущен: {img_path}. Ошибка: {e}")
            return None