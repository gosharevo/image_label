# src/ml/predictor.py

import torch
import os
import cv2
import numpy as np
import hashlib
from torch import nn
from PyQt5.QtCore import QObject, pyqtSignal
from typing import List, Tuple, Optional
from loguru import logger
from pathlib import Path
from functools import partial

from .model import ViTOrdinal
from ..utils.config import MODEL_PATH, NUM_CLASSES, GRAD_CAM_DIR, DROP_PATH_RATE
from .dataset import InferenceDataset
from .transforms import val_transforms


# --- Класс-обертка и вспомогательные функции (без изменений) ---
class ViTWithAttention(nn.Module):
    # ... (код класса ViTWithAttention без изменений) ...
    def __init__(self, vit_model: ViTOrdinal):
        super().__init__()
        self.model = vit_model
        self.attention_maps = []
        for i, block in enumerate(self.model.backbone.blocks):
            block.attn.forward = partial(self.attention_block_forward, block.attn, block_index=i)

    def attention_block_forward(self, attention_module, x, block_index):
        B, N, C = x.shape
        qkv = attention_module.qkv(x).reshape(B, N, 3, attention_module.num_heads,
                                              C // attention_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * attention_module.scale
        attn = attn.softmax(dim=-1)
        if not self.training:
            while len(self.attention_maps) <= block_index: self.attention_maps.append(None)
            self.attention_maps[block_index] = attn.detach()
        attn = attention_module.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attention_module.proj(x);
        x = attention_module.proj_drop(x)
        return x

    def forward(self, x):
        self.attention_maps = [];
        return self.model(x)

    def get_attention_maps(self):
        return self.attention_maps


def attention_rollout(attention_maps: List[torch.Tensor]) -> torch.Tensor:
    # ... (код функции attention_rollout без изменений) ...
    result = torch.eye(attention_maps[0].size(-1), device=attention_maps[0].device)
    for attn in attention_maps:
        attn_fused = attn.mean(dim=1);
        I = torch.eye(attn_fused.size(-1), device=attn_fused.device)
        attn_fused = attn_fused + I;
        attn_fused = attn_fused / (attn_fused.sum(dim=-1, keepdim=True) + 1e-8)
        result = torch.matmul(attn_fused, result)
    return result[0, 0, 1:]


def ordinal_logits_to_class(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # ... (код функции ordinal_logits_to_class без изменений) ...
    probabilities = torch.sigmoid(logits);
    predicted_levels = (probabilities > 0.5).sum(dim=1)
    predicted_class = (1 + predicted_levels).long();
    ones = torch.ones_like(probabilities[:, :1])
    zeros = torch.zeros_like(probabilities[:, :1]);
    probs_extended = torch.cat([ones, probabilities, zeros], dim=1)
    class_probs = probs_extended[:, :-1] - probs_extended[:, 1:]
    confidence = class_probs.gather(1, (predicted_class - 1).unsqueeze(1)).squeeze()
    return predicted_class, confidence


# --- НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ВИЗУАЛИЗАЦИИ ---
def create_flashlight_effect(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Применяет эффект "фонарика", используя тепловую карту как маску.

    Args:
        image (np.ndarray): Исходное BGR изображение (uint8).
        heatmap (np.ndarray): Нормализованная тепловая карта (float, 0.0-1.0).

    Returns:
        np.ndarray: Изображение с примененным эффектом.
    """
    # Изменяем размер тепловой карты до размера изображения
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Добавляем измерение для канала, чтобы умножать (H, W, 1) на (H, W, 3)
    heatmap_mask = heatmap_resized[..., np.newaxis]

    # Умножаем каждый канал изображения на маску.
    # Сначала конвертируем изображение в float для корректного умножения.
    flashlight_img = image.astype(np.float32) * heatmap_mask

    # Конвертируем обратно в uint8 для сохранения
    return flashlight_img.astype(np.uint8)


class Predictor(QObject):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)

    def __init__(self, image_paths: List[str]):
        super().__init__()
        self.image_paths = image_paths

    def run(self):
        logger.info(f"Запуск инференса для {len(self.image_paths)} изображений.")
        predictions = {}

        if not os.path.exists(MODEL_PATH):
            logger.error(f"Модель для инференса не найдена: {MODEL_PATH}")
            self.finished.emit({})
            return

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.progress.emit(f"Инференс на {device.type.upper()}...")

            base_model = ViTOrdinal(num_classes=NUM_CLASSES, pretrained=False, drop_path_rate=DROP_PATH_RATE).to(device)
            base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

            model = ViTWithAttention(base_model).to(device)
            model.eval()
            logger.info("Модель успешно загружена и обернута для извлечения внимания.")

            dataset = InferenceDataset(self.image_paths, transform=val_transforms)
            for i, (image_tensor, path) in enumerate(dataset):
                if image_tensor is None or path is None:
                    continue

                self.progress.emit(f"Обработка {i + 1}/{len(dataset)}: {os.path.basename(path)}")
                input_tensor = image_tensor.to(device).unsqueeze(0)

                logits = model(input_tensor)
                pred_class, confidence = ordinal_logits_to_class(logits)
                predictions[path] = [int(pred_class.item()), float(confidence.item())]

                try:
                    attn_maps = model.get_attention_maps()
                    if not attn_maps or any(m is None for m in attn_maps):
                        raise RuntimeError("Не удалось получить карты внимания из модели.")

                    rollout = attention_rollout(attn_maps)

                    # --- ИЗМЕНЕНИЕ В ЛОГИКЕ ВИЗУАЛИЗАЦИИ ---

                    # 1. Нормализуем тепловую карту
                    num_patches = rollout.shape[0]
                    h = w = int(np.sqrt(num_patches))
                    if h * w != num_patches: raise ValueError(f"Число патчей ({num_patches}) не кв.")
                    heatmap = rollout.reshape(h, w).cpu().numpy()
                    heatmap = np.maximum(heatmap, 0)
                    heatmap /= (np.max(heatmap) + 1e-8)  # Нормализация в диапазон [0, 1]

                    # 2. Загружаем оригинальное изображение
                    original_img = cv2.imread(path)
                    if original_img is None:
                        logger.warning(f"Не удалось прочитать изображение: {path}")
                        continue

                    # 3. Создаем эффект "фонарика"
                    flashlight_image = create_flashlight_effect(original_img, heatmap)

                    # 4. Сохраняем результат
                    GRAD_CAM_DIR.mkdir(parents=True, exist_ok=True)
                    path_hash = hashlib.sha256(path.encode()).hexdigest()
                    save_path = GRAD_CAM_DIR / f"{path_hash}.jpg"
                    cv2.imwrite(str(save_path), flashlight_image)  # Сохраняем новое изображение

                except Exception as e:
                    logger.warning(f"Не удалось сгенерировать attention map для {path}: {e}", exc_info=True)

            logger.success("Инференс завершён.")
            self.finished.emit(predictions)

        except Exception as e:
            logger.critical(f"Критическая ошибка во время инференса: {e}", exc_info=True)
            self.finished.emit({})

