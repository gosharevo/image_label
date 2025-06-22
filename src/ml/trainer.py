# src/ml/trainer.py (версия без wandb)

# --- Системные и ML библиотеки ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from pathlib import Path

# --- Логирование и GUI ---
from loguru import logger
from PyQt5.QtCore import QObject, pyqtSignal

# --- Компоненты проекта ---
from .model import ViTOrdinal
from ..utils.config import (NUM_CLASSES, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, MODEL_PATH,
                            VAL_SPLIT_SIZE, RANDOM_STATE, DROP_PATH_RATE, WARMUP_EPOCHS)
from .dataset import ImageLabelDataset, collate_fn
from .transforms import train_transforms, val_transforms


# --- Вспомогательные функции ---

def ordinal_logits_to_class(logits: torch.Tensor) -> torch.Tensor:
    """Преобразует порядковые логиты в предсказанный класс (1-индексированный)."""
    probabilities = torch.sigmoid(logits)
    predicted_levels = (probabilities > 0.5).sum(dim=1)
    return (1 + predicted_levels).long()


def label_to_ordinal_target(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Преобразует 0-индексированные метки в бинарные цели."""
    target = torch.zeros(labels.size(0), num_classes - 1, dtype=torch.float32, device=labels.device)
    for i, label in enumerate(labels):
        if label > 0:
            target[i, :label] = 1.0
    return target


def calculate_mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Вычисляет Mean Absolute Error."""
    return torch.abs(preds - targets).float().mean().item()


class Trainer(QObject):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, dataset: List[Tuple[str, int]]):
        super().__init__()
        self.dataset_list = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.results_dir = Path("results") / self.run_timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Разделяет данные на train/val и создает DataLoader'ы."""
        self.progress.emit("Разделение данных на обучающий и валидационный наборы...")

        paths = [item[0] for item in self.dataset_list]
        labels = [item[1] for item in self.dataset_list]

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths, labels, test_size=VAL_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=labels)

        train_list = list(zip(train_paths, train_labels))
        val_list = list(zip(val_paths, val_labels))

        logger.info(f"Размер обучающего набора: {len(train_list)}")
        logger.info(f"Размер валидационного набора: {len(val_list)}")
        from collections import Counter
        logger.info(f"✅ Классы в трейне: {Counter(train_labels)}")
        logger.info(f"✅ Классы в валидации: {Counter(val_labels)}")

        train_dataset = ImageLabelDataset(train_list, transform=train_transforms)
        val_dataset = ImageLabelDataset(val_list, transform=val_transforms)

        num_workers = 2 if os.name != 'nt' and self.device.type == 'cuda' else 0
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE * 2,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
        return train_loader, val_loader

    def _calculate_ordinal_pos_weights(self) -> torch.Tensor:
        # ... (реализация без изменений) ...
        raw_labels = torch.tensor([label - 1 for _, label in self.dataset_list])
        ordinal_targets = label_to_ordinal_target(raw_labels, NUM_CLASSES)
        num_positives = torch.sum(ordinal_targets, dim=0)
        num_negatives = len(self.dataset_list) - num_positives
        pos_weight = num_negatives / (num_positives + 1e-8)
        logger.info(f"pos_weight = {[round(float(w), 2) for w in pos_weight.cpu()]}")

        return pos_weight

    def evaluate(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Запускает цикл оценки на валидационном наборе."""
        model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, labels_1_indexed in loader:
                if inputs is None: continue
                inputs, labels_1_indexed = inputs.to(self.device), labels_1_indexed.to(self.device)
                logits = model(inputs)

                loss = criterion(logits, label_to_ordinal_target(labels_1_indexed - 1, NUM_CLASSES))
                total_loss += loss.item()

                all_preds.append(ordinal_logits_to_class(logits))
                all_targets.append(labels_1_indexed)

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
        mae = calculate_mae(torch.cat(all_preds), torch.cat(all_targets))

        return {'loss': avg_loss, 'mae': mae}

    def _plot_metrics(self, history: Dict[str, List[float]]):
        """Сохраняет графики метрик обучения."""
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # График Loss
        ax1.plot(history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params('y', colors='b')
        ax1.grid(True)

        # График MAE на второй оси Y
        ax2 = ax1.twinx()
        ax2.plot(history['val_mae'], 'g--', label='Validation MAE')
        ax2.set_ylabel('MAE', color='g')
        ax2.tick_params('y', colors='g')

        fig.tight_layout()
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        plt.title('Training and Validation Metrics')

        plot_path = self.results_dir / "training_metrics.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Графики обучения сохранены в: {plot_path}")

    def run(self):
        try:
            logger.info(f"Начало процесса обучения. Результаты будут в: {self.results_dir}")
            logger.info(f"Обучение на устройстве: {self.device.type.upper()}")

            # --- Подготовка данных и модели ---
            train_loader, val_loader = self._prepare_data()
            model = ViTOrdinal(
                num_classes=NUM_CLASSES, pretrained=True, drop_path_rate=DROP_PATH_RATE
            ).to(self.device)

            pos_weights = self._calculate_ordinal_pos_weights().to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS,
                                                                  eta_min=1e-6)
            warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)

            # --- Цикл обучения ---
            best_val_mae = float('inf')
            history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
            no_result_count = 0

            for epoch in range(NUM_EPOCHS):
                model.train()
                running_loss = 0.0
                num_batches = len(train_loader)

                self.progress.emit(f"Эпоха {epoch + 1}/{NUM_EPOCHS}...")
                for batch_index, (inputs, labels_1_indexed) in enumerate(train_loader, 1):
                    if inputs is None:
                        continue
                    self.progress.emit(f"Эпоха {epoch + 1}/{NUM_EPOCHS}, батч {batch_index}/{len(train_loader)}")

                    inputs = inputs.to(self.device)
                    ordinal_targets = label_to_ordinal_target(labels_1_indexed - 1, NUM_CLASSES).to(self.device)

                    optimizer.zero_grad()
                    logits = model(inputs)
                    loss = criterion(logits, ordinal_targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                # --- Валидация ---
                val_metrics = self.evaluate(model, val_loader, criterion)
                avg_train_loss = running_loss / num_batches if num_batches > 0 else 0

                # --- Логирование и сохранение истории ---
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(val_metrics['loss'])
                history['val_mae'].append(val_metrics['mae'])

                logger.info(
                    f"Эпоха {epoch + 1}: Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, Val MAE={val_metrics['mae']:.4f}, "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}"
                )

                # --- Обновление LR и сохранение лучшей модели ---
                if epoch < WARMUP_EPOCHS:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step()

                if val_metrics['mae'] < best_val_mae:
                    best_val_mae = val_metrics['mae']
                    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), str(MODEL_PATH))
                    logger.success(f"Новая лучшая модель сохранена! Val MAE: {best_val_mae:.4f}")
                    no_result_count = 0
                else:
                    no_result_count += 1
                    if no_result_count == 3:
                        logger.info('За 3 эпохи ничего не улучшилось, остановка...')
                        break

            # --- Финализация ---
            self._plot_metrics(history)
            logger.success(f"Обучение завершено. Лучшая модель сохранена в {MODEL_PATH} с Val MAE: {best_val_mae:.4f}")
            self.finished.emit(True, f"Обучение завершено. Лучшая модель с MAE={best_val_mae:.2f} сохранена.")

        except Exception as e:
            logger.critical(f"Критическая ошибка в процессе обучения: {e}", exc_info=True)
            self.finished.emit(False, f"Ошибка обучения: {e}")

