import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Callable
from loguru import logger

class ImageLabelDataset(Dataset):
    """Кастомный датасет для PyTorch."""
    def __init__(self, data: List[Tuple[str, int]], transform: Callable):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label - 1
        except Exception:
            logger.error(f"Не удалось прочитать или обработать файл {img_path}. Пропускаю.", exc_info=True)
            return None, None

def collate_fn(batch):
    """
    Функция для DataLoader, которая отфильтровывает "битые" данные (None),
    возвращаемые из __getitem__ в случае ошибки.
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

class InferenceDataset(Dataset):
    """Датасет для инференса, работает без меток."""
    def __init__(self, image_paths: List[str], transform: Callable):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception:
            logger.error(f"Не удалось прочитать файл для инференса: {img_path}", exc_info=True)
            return None, None