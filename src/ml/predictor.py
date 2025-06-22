import torch
import os
import cv2
import numpy as np
import hashlib
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch import nn

from PyQt5.QtCore import QObject, pyqtSignal
from typing import List, Dict, Tuple
from loguru import logger

from ..utils.config import MODEL_PATH, NUM_CLASSES, BATCH_SIZE, GRAD_CAM_DIR, GRAD_MODE
from .dataset import InferenceDataset, collate_fn
from .transforms import val_transforms



class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        h1 = target_layer.register_forward_hook(self._forward_hook)
        h2 = target_layer.register_backward_hook(self._backward_hook)
        self.handles = [h1, h2]

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def generate_cam(self, input_tensor: torch.Tensor, original_image_path: str, class_idx: int, mode: str = "fade"):
        self.model.eval()
        logits = self.model(input_tensor.unsqueeze(0))
        self.model.zero_grad()

        if class_idx <= 1:
            target_logit_idx = 0
        else:
            target_logit_idx = min(class_idx - 2, logits.shape[1] - 1)

        score = logits[:, target_logit_idx].sum()
        score.backward()

        if self.gradients is None or self.activations is None:
            return

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1.0

        try:
            img = cv2.imread(original_image_path)
            if img is None:
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

            if mode == "fade":
                heatmap_3ch = np.expand_dims(heatmap, axis=-1)
                blended = (img.astype(np.float32) * heatmap_3ch).astype(np.uint8)
            elif mode == "jet":
                heatmap_uint8 = np.uint8(255 * heatmap)
                jet_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                blended = cv2.addWeighted(jet_map, 0.4, img, 0.6, 0).astype(np.uint8)
            else:
                blended = img  # fallback

            blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

            GRAD_CAM_DIR.mkdir(parents=True, exist_ok=True)
            path_hash = hashlib.sha256(original_image_path.encode()).hexdigest()
            save_path = GRAD_CAM_DIR / f"{path_hash}.jpg"
            cv2.imwrite(str(save_path), blended)

        except Exception:
            return


def ordinal_logits_to_class(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.sigmoid(logits)
    predicted_levels = (probabilities > 0.5).sum(dim=1)
    predicted_class = (1 + predicted_levels).long()
    ones = torch.ones_like(probabilities[:, :1])
    zeros = torch.zeros_like(probabilities[:, :1])
    probs_extended = torch.cat([ones, probabilities, zeros], dim=1)
    class_probs = probs_extended[:, :-1] - probs_extended[:, 1:]
    confidence = class_probs.gather(1, (predicted_class - 1).unsqueeze(1)).squeeze()
    return predicted_class, confidence


class Predictor(QObject):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)

    def __init__(self, image_paths: List[str]):
        super().__init__()
        self.image_paths = image_paths

    def run(self):
        logger.info(f"Запуск инференса с генерацией Grad-CAM для {len(self.image_paths)} изображений.")
        predictions = {}

        if not os.path.exists(MODEL_PATH):
            logger.error("Модель для инференса не найдена.")
            self.finished.emit({})
            return

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.progress.emit(f"Инференс на {device.type.upper()}...")

            model = resnet18()
            model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES - 1)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model = model.to(device)
            model.eval()

            # Инициализация Grad-CAM
            grad_cam = GradCAM(model, model.layer4)

            dataset = InferenceDataset(self.image_paths, transform=val_transforms)

            for i, (image_tensor, path) in enumerate(dataset):
                if image_tensor is None or path is None:
                    continue

                self.progress.emit(f"Обработка {i + 1}/{len(dataset)}: {os.path.basename(path)}")

                input_tensor = image_tensor.to(device)

                with torch.no_grad():
                    logits = model(input_tensor.unsqueeze(0))

                pred_class, confidence = ordinal_logits_to_class(logits)
                pred_class_item = int(pred_class.item())
                confidence_item = float(confidence.item())

                predictions[path] = [pred_class_item, confidence_item]

                # Генерация Grad-CAM для этого изображения
                try:
                    # Создаем копию тензора для Grad-CAM
                    grad_cam_tensor = input_tensor.clone().detach()
                    grad_cam_tensor.requires_grad = True
                    grad_cam.generate_cam(grad_cam_tensor, path, pred_class_item, GRAD_MODE)
                except Exception as e:
                    logger.error(f"Не удалось сгенерировать Grad-CAM для {path}: {e}")

            # Удаляем хуки после завершения
            grad_cam.remove_hooks()

            logger.success(f"Инференс и генерация Grad-CAM завершены.")
            self.finished.emit(predictions)

        except Exception as e:
            logger.critical(f"Критическая ошибка во время инференса: {e}", exc_info=True)
            self.finished.emit({})
