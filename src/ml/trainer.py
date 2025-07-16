# src/ml/trainer.py
import shutil

# --- Системные и ML библиотеки ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel

# --- Научные вычисления и метрики ---
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from ..utils.config import MODEL_PATH
# --- Утилиты и конфигурация ---
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter
from loguru import logger
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter

# --- GUI ---
from PyQt5.QtCore import QObject, pyqtSignal

# --- Компоненты проекта ---
from .model import ViTOrdinal
from .utils import TrainingConfig
from .dataset import ImageLabelDataset  # Dataset теперь сам будет делать всю работу
from .transforms import train_transforms, val_transforms


# --- Вспомогательные функции и классы ---

def collate_fn(batch: List[Optional[Tuple]]) -> Optional[Tuple]:
    """
    Отказоустойчивая collate_fn. Она получает список результатов от __getitem__.
    Если какой-то элемент - None (из-за ошибки чтения файла), он отфильтровывается.
    """
    # 1. Отфильтровываем все None значения, которые вернул Dataset
    batch = [item for item in batch if item is not None]

    # 2. Если после фильтрации весь батч оказался пустым, возвращаем None
    if not batch:
        return None, None

    # 3. Если остались валидные данные, распаковываем их и собираем в тензоры
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)


class OrdinalLoss(nn.Module):
    def __init__(self, config: TrainingConfig, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}
        bce_loss = self.bce_loss(logits, targets)
        # Focal Loss модуляция
        p = torch.sigmoid(logits)
        focal_weight = torch.where(targets > 0.5, (1 - p) ** 2.0, p ** 2.0)
        bce_loss = focal_weight * bce_loss
        losses['ordinal'] = bce_loss.mean()
        losses['total'] = losses['ordinal']
        return losses


class OrdinalMetricsCalculator:
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.latest_cm = None

    def calculate_all(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        preds_np, targets_np = preds.cpu().numpy(), targets.cpu().numpy()
        metrics = {
            'mae': np.abs(preds_np - targets_np).mean(),
            'rmse': np.sqrt(((preds_np - targets_np) ** 2).mean()),
            'accuracy': (preds_np == targets_np).mean()
        }
        labels = list(range(1, self.num_classes + 1))
        metrics['qwk'] = cohen_kappa_score(targets_np, preds_np, weights='quadratic', labels=labels)
        self.latest_cm = confusion_matrix(targets_np, preds_np, labels=labels)
        return metrics

    def plot_dashboard(self, save_path: Path):
        if self.latest_cm is None:
            return
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle('Финальная матрица ошибок', fontsize=16)
        cm_normalized = self.latest_cm.astype('float') / self.latest_cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax1,
                    xticklabels=self.class_names, yticklabels=self.class_names)
        ax1.set_xlabel('Предсказанный класс')
        ax1.set_ylabel('Истинный класс')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300)
        plt.close()


def enforce_monotonicity(logits: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(torch.nn.functional.relu(logits), dims=[1]), dim=1), dims=[1])


def ordinal_logits_to_class(logits: torch.Tensor) -> torch.Tensor:
    return (1 + (torch.sigmoid(logits) > 0.5).sum(dim=1)).long()


def label_to_ordinal_target(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    target = torch.zeros(labels.size(0), num_classes - 1, device=labels.device)
    for i, label in enumerate(labels):
        if label > 0:
            target[i, :label] = 1.0
    return target * (1.0 - smoothing) + smoothing / 2


# === Основной класс тренера (ФИНАЛЬНАЯ ВЕРСИЯ) ===
class Trainer(QObject):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, dataset: List[Tuple[str, int]]):
        super().__init__()
        self.full_dataset_list = dataset
        self.config = TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.results_dir = Path("results") / self.run_timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()
        self.tb_writer = SummaryWriter(log_dir=str(self.results_dir / 'tensorboard'))
        self.class_names = [f"Уровень {i}" for i in range(1, self.config.num_classes + 1)]
        self.metrics_calculator = OrdinalMetricsCalculator(self.config.num_classes, self.class_names)
        self.scaler = GradScaler(enabled=self.device.type == 'cuda')

    def _save_config(self):
        with open(self.results_dir / "config.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(self.config.to_dict(), f, allow_unicode=True, sort_keys=False)

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        self.progress.emit("Разделение данных и подготовка загрузчиков...")
        paths = [item[0] for item in self.full_dataset_list]
        labels = [item[1] for item in self.full_dataset_list]

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths, labels, test_size=self.config.val_split_size,
            random_state=self.config.random_state, stratify=labels)

        train_dataset = ImageLabelDataset(list(zip(train_paths, train_labels)), transform=train_transforms)
        val_dataset = ImageLabelDataset(list(zip(val_paths, val_labels)), transform=val_transforms)
        logger.info(f"Обучающий набор: {len(train_dataset)} | Валидационный набор: {len(val_dataset)}")

        class_counts = Counter(train_labels)
        weights = 1. / torch.tensor([class_counts[i] for i in sorted(class_counts)], dtype=torch.float)
        sample_weights = weights[[l - 1 for l in train_labels]]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        num_workers = 4 if self.device.type == 'cuda' and os.name != 'nt' else 0
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, sampler=sampler,
                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size * 2, num_workers=num_workers,
                                collate_fn=collate_fn)
        return train_loader, val_loader

    def _calculate_pos_weights(self) -> torch.Tensor:
        all_labels = [item[1] for item in self.full_dataset_list]
        raw_labels = torch.tensor([label - 1 for label in all_labels])
        ordinal_targets = label_to_ordinal_target(raw_labels, self.config.num_classes, 0.0)
        num_pos = torch.sum(ordinal_targets, dim=0)
        num_neg = len(all_labels) - num_pos
        return torch.clamp(num_neg / (num_pos + 1e-8), min=0.5, max=10.0)

    def _log_metrics(self, metrics: Dict, epoch: int, stage: str):
        for k, v in metrics.items():
            if isinstance(v, (int, float)): self.tb_writer.add_scalar(f'{stage}/{k}', v, epoch)
        logger.info(
            f"Этап '{stage.upper()}', эпоха {epoch + 1}: MAE={metrics.get('mae', -1):.4f}, QWK={metrics.get('qwk', -1):.4f}, Потеря={metrics.get('loss_total', -1):.4f}")

    def evaluate(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict:
        model.eval()
        all_logits, all_targets_1_indexed = [], []
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                if inputs is None or labels is None:
                    continue

                inputs = inputs.to(self.device)
                labels_1_indexed = labels.to(self.device, dtype=torch.long)

                with autocast(enabled=self.device.type == 'cuda'):
                    logits = model(inputs)
                    logits = enforce_monotonicity(logits)
                    ordinal_targets = label_to_ordinal_target(labels_1_indexed - 1, self.config.num_classes, 0.0)
                    total_loss += criterion(logits, ordinal_targets)['total'].item()

                all_logits.append(logits.cpu())
                all_targets_1_indexed.append(labels_1_indexed.cpu())

        metrics = self.metrics_calculator.calculate_all(ordinal_logits_to_class(torch.cat(all_logits)),
                                                        torch.cat(all_targets_1_indexed))
        metrics['loss_total'] = total_loss / len(loader) if len(loader) > 0 else 0
        return metrics

    def run(self):
        try:
            logger.info(f"Запуск процесса обучения. Результаты в: {self.results_dir}")
            train_loader, val_loader = self._prepare_data()
            model = ViTOrdinal(num_classes=self.config.num_classes, pretrained=True,
                               drop_path_rate=self.config.drop_path_rate).to(self.device)
            ema_model = AveragedModel(model, avg_fn=lambda avg, m, n: 0.999 * avg + 0.001 * m)
            pos_weights = self._calculate_pos_weights().to(self.device)
            criterion = OrdinalLoss(self.config, pos_weights)
            optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)
            warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                                           total_iters=self.config.warmup_epochs)
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=self.config.num_epochs - self.config.warmup_epochs,
                                                                  eta_min=1e-6)
            best_val_mae, epochs_no_improve = float('inf'), 0

            for epoch in range(self.config.num_epochs):
                self.progress.emit(f"Эпоха {epoch + 1}/{self.config.num_epochs}...")
                model.train()
                pbar = tqdm(train_loader, desc=f"Обучение (Эпоха {epoch + 1})", leave=False)

                for inputs, labels in pbar:
                    if inputs is None or labels is None:
                        continue

                    inputs = inputs.to(self.device)
                    labels_1_indexed = labels.to(self.device, dtype=torch.long)
                    ordinal_targets = label_to_ordinal_target(labels_1_indexed - 1, self.config.num_classes,
                                                              self.config.label_smoothing)

                    optimizer.zero_grad()
                    with autocast(enabled=self.device.type == 'cuda'):
                        logits = model(inputs)
                        logits = enforce_monotonicity(logits)
                        loss = criterion(logits, ordinal_targets)['total']

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_val)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    ema_model.update_parameters(model)
                    pbar.set_postfix(loss=loss.item())

                val_metrics = self.evaluate(ema_model.module, val_loader, criterion)
                self._log_metrics(val_metrics, epoch, 'validation')

                if epoch < self.config.warmup_epochs:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step()

                if val_metrics['mae'] < best_val_mae:
                    best_val_mae = val_metrics['mae']
                    torch.save({
                        'model_state_dict': ema_model.module.state_dict(),
                        'config': self.config.to_dict(),
                        'thresholds': torch.full((self.config.num_classes - 1,), 0.5)
                    }, self.results_dir / "best_model.pth")
                    shutil.copy(self.results_dir / "best_model.pth", MODEL_PATH)
                    logger.success(f"Новая лучшая модель с Val MAE: {best_val_mae:.4f}. Сохранено.")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.config.early_stopping_patience:
                        logger.warning("Ранняя остановка.")
                        break

            logger.info("Обучение завершено. Финальный отчет сгенерирован.")
            self.metrics_calculator.plot_dashboard(self.results_dir / "final_dashboard.png")
            self.finished.emit(True, f"Обучение завершено. Итоговая MAE на валидации: {best_val_mae:.2f}")

        except Exception as e:
            logger.critical(f"Критическая ошибка: {e}", exc_info=True)
            self.finished.emit(False, f"Ошибка обучения: {e}")
        finally:
            self.tb_writer.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


