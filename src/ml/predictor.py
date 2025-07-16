# src/ml/predictor.py

# --- Базовые и системные библиотеки ---
import torch
import os
import cv2
import json
import numpy as np
import hashlib
from torch import nn
from torch.utils.data import DataLoader
from functools import partial
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
from pathlib import Path

# --- Компоненты проекта ---
from .model import ViTOrdinal
from .utils import TrainingConfig, load_model_for_inference
from .dataset import InferenceDataset
from .transforms import val_transforms


# --- Вспомогательные функции и классы ---

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None
    images, paths = zip(*batch)
    return torch.stack(images), paths


@dataclass
class PredictionResult:
    image_path: str;
    predicted_class: int;
    confidence: float
    uncertainty: float;
    visualization_path: Optional[Path] = None;
    from_cache: bool = False


def ordinal_logits_to_results(logits: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probabilities = 1 / (1 + np.exp(-logits));
    predicted_levels = (probabilities > thresholds).sum(axis=1)
    predicted_class = (1 + predicted_levels).astype(np.int64);
    ones = np.ones((probabilities.shape[0], 1))
    zeros = np.zeros((probabilities.shape[0], 1));
    probs_extended = np.concatenate([ones, probabilities, zeros], axis=1)
    class_probs = probs_extended[:, :-1] - probs_extended[:, 1:]
    confidence = np.take_along_axis(class_probs, (predicted_class - 1)[:, np.newaxis], axis=1).squeeze()
    return predicted_class, confidence


def create_diagnostic_image(image_path: str, heatmap: np.ndarray, result: PredictionResult, save_path: Path):
    original_bgr = cv2.imread(image_path)
    if original_bgr is None: logger.warning(f"Не удалось прочитать изображение: {image_path}"); return
    h, w, _ = original_bgr.shape
    heatmap_resized = cv2.resize(heatmap, (w, h));
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0);
    info_panel = np.zeros((h, w, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    texts = [f"Предсказание: {result.predicted_class}", f"Уверенность: {result.confidence:.2%}",
             f"Неопределенность: {result.uncertainty:.3f}"]
    for i, text in enumerate(texts): cv2.putText(info_panel, text, (10, 30 + i * 30), font, 0.7, (255, 255, 255), 2)
    top_row = np.hstack((original_bgr, heatmap_colored));
    bottom_row = np.hstack((blended, info_panel))
    diagnostic_img = np.vstack((top_row, bottom_row));
    cv2.imwrite(str(save_path), diagnostic_img)
    result.visualization_path = save_path;
    logger.debug(f"Диагностическое изображение создано: {save_path}")


# --- САМЫЙ НАДЕЖНЫЙ СПОСОБ ПОЛУЧЕНИЯ ВНИМАНИЯ ---
class ViTWithAttention(nn.Module):
    """
    Класс-обертка, который гарантированно перехватывает карты внимания
    путем прямого "патчинга" forward-метода модулей внимания.
    """

    def __init__(self, vit_model: ViTOrdinal):
        super().__init__()
        self.model = vit_model
        self.attention_maps = []
        # Мы заменяем оригинальный метод forward на наш собственный,
        # который делает то же самое, но сохраняет карты внимания.
        for block in self.model.backbone.blocks:
            block.attn.forward = partial(self.attention_block_forward, block.attn)

    def attention_block_forward(self, attention_module, x):
        """Патч, заменяющий оригинальный forward. Гарантированно захватывает карты внимания."""
        B, N, C = x.shape
        qkv = attention_module.qkv(x).reshape(B, N, 3, attention_module.num_heads,
                                              C // attention_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * attention_module.scale
        attn = attn.softmax(dim=-1)

        # Сохраняем карты внимания
        self.attention_maps.append(attn.detach())

        attn = attention_module.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attention_module.proj(x)
        x = attention_module.proj_drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.attention_maps = []  # Очищаем перед каждым прогоном
        return self.model(x)


def attention_rollout(attention_maps: List[torch.Tensor]) -> np.ndarray:
    """Агрегирует карты внимания. Теперь с защитой от пустого списка."""
    if not attention_maps:
        logger.warning("Карты внимания не были сгенерированы. Визуализация невозможна.")
        return np.array([])  # Возвращаем пустой массив, чтобы не было ошибки

    result = torch.eye(attention_maps[0].shape[-1], device=attention_maps[0].device)
    for attn in attention_maps:
        attn = attn.mean(dim=1)
        residual_attn = attn + torch.eye(attn.shape[-1], device=attn.device)
        agg_attn = residual_attn / residual_attn.sum(dim=-1, keepdim=True)
        result = torch.matmul(agg_attn, result)
    return result[:, 0, 1:].cpu().numpy()


class InferenceJob(QRunnable):
    def __init__(self, predictor_instance: 'Predictor', image_paths: List[str], use_uncertainty: bool, mc_runs: int):
        super().__init__();
        self.predictor, self.image_paths, self.use_uncertainty, self.mc_runs = predictor_instance, image_paths, use_uncertainty, mc_runs

    def run(self):
        try:
            results = self.predictor._perform_inference(self.image_paths, self.use_uncertainty, self.mc_runs)
        except Exception as e:
            logger.critical(f"Критическая ошибка в потоке инференса: {e}", exc_info=True); results = {}
        finally:
            self.predictor.finished.emit(results)
            if torch.cuda.is_available(): torch.cuda.empty_cache()


class Predictor(QObject):
    finished = pyqtSignal(dict);
    progress = pyqtSignal(str)

    def __init__(self, model_path_str: str = "models/best_model.pth"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path_str)
        self.cache_dir = Path("cache");
        self.viz_dir = self.cache_dir / "visualizations"
        self.cache_dir.mkdir(exist_ok=True);
        self.viz_dir.mkdir(exist_ok=True)
        self.base_model = None;
        self.attention_model = None
        self.config = None;
        self.thresholds = None
        self.thread_pool = QThreadPool();
        self.thread_pool.setMaxThreadCount(1)
        self._initialize_engine()

    def _initialize_engine(self):
        try:
            self.progress.emit("Инициализация движка инференса...")
            self.base_model, self.config, thresholds_tensor = load_model_for_inference(self.model_path, self.device)
            self.thresholds = thresholds_tensor.cpu().numpy()
            self.attention_model = ViTWithAttention(self.base_model)
            logger.success("Движок инференса (PyTorch) успешно инициализирован.")
            self.progress.emit("Движок готов к работе.")
        except Exception as e:
            logger.critical(f"Не удалось инициализировать движок инференса: {e}", exc_info=True)

    def run(self, image_paths: List[str], use_uncertainty: bool = True, mc_runs: int = 10):
        if self.base_model is None: return
        job = InferenceJob(self, image_paths, use_uncertainty, mc_runs)
        self.thread_pool.start(job)

    def _perform_inference(self, image_paths: List[str], use_uncertainty: bool, mc_runs: int) -> Dict[
        str, PredictionResult]:
        final_results = {};
        paths_to_process = self._check_cache(image_paths, final_results)
        if not paths_to_process: return final_results

        dataset = InferenceDataset(paths_to_process, transform=val_transforms)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn)
        num_runs = mc_runs if use_uncertainty else 1

        from tqdm import tqdm
        pbar = tqdm(loader, desc="Инференс и анализ", leave=False)
        for images, paths_in_batch in pbar:
            if images is None: continue

            images = images.to(self.device)

            # 1. Оценка неопределенности на быстрой, базовой модели
            self.base_model.train() if use_uncertainty else self.base_model.eval()
            mc_logits = []
            for _ in range(num_runs):
                with torch.no_grad(): mc_logits.append(self.base_model(images).cpu().numpy())

            # 2. Карты внимания на обернутой модели (только 1 проход)
            self.attention_model.eval()
            with torch.no_grad():
                _ = self.attention_model(images)
            rollouts = attention_rollout(self.attention_model.attention_maps)

            # 3. Обработка результатов
            avg_logits = np.stack(mc_logits).mean(axis=0)
            final_preds, final_confidences = ordinal_logits_to_results(avg_logits, self.thresholds)
            uncertainties = np.std(np.array([ordinal_logits_to_results(l, self.thresholds)[0] for l in mc_logits]),
                                   axis=0) if use_uncertainty and num_runs > 1 else np.zeros_like(final_preds,
                                                                                                  dtype=float)
            final_preds, final_confidences, uncertainties = np.atleast_1d(final_preds), np.atleast_1d(
                final_confidences), np.atleast_1d(uncertainties)

            for i, path in enumerate(paths_in_batch):
                result = PredictionResult(image_path=path, predicted_class=final_preds[i].item(),
                                          confidence=final_confidences[i].item(), uncertainty=uncertainties[i].item())

                # Защита от всех возможных сбоев при генерации карты
                if rollouts.size > 0 and rollouts.shape[0] == len(paths_in_batch):
                    num_patches = rollouts.shape[-1]
                    grid_size = int(np.sqrt(num_patches))
                    if grid_size * grid_size == num_patches:
                        heatmap = rollouts[i].reshape(grid_size, grid_size)
                        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                        path_hash = hashlib.sha256(path.encode()).hexdigest()
                        viz_save_path = self.viz_dir / f"diag_{path_hash}.jpg"
                        create_diagnostic_image(path, heatmap, result, viz_save_path)

                final_results[path] = result;
                self._cache_result(path, result)

        self.base_model.eval()
        return final_results

    def _check_cache(self, image_paths: List[str], final_results: Dict) -> List[str]:
        paths_to_process = []
        for path in image_paths:
            path_hash = hashlib.sha256(path.encode()).hexdigest();
            cache_file = self.cache_dir / f"{path_hash}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    result = PredictionResult(**cached_data)
                    result.from_cache = True;
                    final_results[path] = result
                except Exception:
                    paths_to_process.append(path)
            else:
                paths_to_process.append(path)
        if paths_to_process: self.progress.emit(
            f"Найдено {len(final_results)} в кэше. Обработка {len(paths_to_process)}...")
        return paths_to_process

    def _cache_result(self, path: str, result: PredictionResult):
        path_hash = hashlib.sha256(path.encode()).hexdigest();
        cache_file = self.cache_dir / f"{path_hash}.json"
        result_dict = result.__dict__
        if isinstance(result_dict['visualization_path'], Path): result_dict['visualization_path'] = str(
            result_dict['visualization_path'])
        with open(cache_file, 'w') as f: json.dump(result_dict, f, indent=2)


