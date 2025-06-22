from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from PyQt5.QtCore import QObject, pyqtSignal
from typing import List, Tuple
from loguru import logger
import matplotlib.pyplot as plt

from ..utils.config import NUM_CLASSES, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, MODEL_PATH
from .dataset import ImageLabelDataset, collate_fn
from .transforms import train_transforms


def label_to_ordinal_target(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Преобразует метку класса (e.g., 2 для 0-индексации) в вектор для порядковой регрессии.
    метка 2 (класс 3) -> [1, 1, 0, 0, 0] (для num_classes=6)
    """
    target = torch.zeros(labels.size(0), num_classes - 1, dtype=torch.float32)
    for i, label in enumerate(labels):
        if label > 0:
            target[i, :label] = 1
    return target


class Trainer(QObject):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, dataset: List[Tuple[str, int]]):
        super().__init__()
        self.dataset_list = dataset

    def _calculate_ordinal_pos_weights(self) -> torch.Tensor:
        """
        Рассчитывает pos_weight для каждого бинарного классификатора в Ordinal Regression.
        Это ключевая техника для борьбы с дисбалансом в этой архитектуре.
        """
        logger.info("Расчет pos_weights для Ordinal Regression...")

        # метки у нас 1-6
        raw_labels = torch.tensor([label for _, label in self.dataset_list])
        if len(raw_labels) == 0:
            return torch.ones(NUM_CLASSES - 1)

        # конвертируем в 0-5
        labels_0_indexed = raw_labels - 1

        ordinal_targets = label_to_ordinal_target(labels_0_indexed, NUM_CLASSES)

        num_positives = torch.sum(ordinal_targets, dim=0)
        num_negatives = len(self.dataset_list) - num_positives

        # Избегаем деления на ноль, если для какого-то вопроса нет позитивных примеров
        # (что маловероятно, но лучше перестраховаться)
        pos_weight = num_negatives / (num_positives + 1e-8)

        logger.info(f"Количество позитивных ответов для вопросов [>1, >2, ...]: {num_positives.tolist()}")
        logger.info(f"Рассчитанные pos_weights: {[f'{w:.2f}' for w in pos_weight]}")

        return pos_weight

    def run(self):
        logger.info("Начало процесса обучения взвешенной Ordinal Regression модели.")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Обучение будет производиться на устройстве: {device.type.upper()}")

            if len(self.dataset_list) < BATCH_SIZE:
                raise ValueError(f"Недостаточно данных для обучения. Нужно хотя бы {BATCH_SIZE} сэмплов.")

            # --- Расчет весов (КЛЮЧЕВОЕ ИЗМЕНЕНИЕ) ---
            pos_weights = self._calculate_ordinal_pos_weights().to(device)

            # --- Подготовка данных ---
            self.progress.emit("Подготовка данных...")
            train_dataset = ImageLabelDataset(self.dataset_list, transform=train_transforms)
            num_workers = 0 if os.name == 'nt' else 2
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True if device.type == 'cuda' else False
            )

            # --- Инициализация модели ---
            self.progress.emit("Инициализация модели ResNet18 для Ordinal Regression...")
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, NUM_CLASSES - 1)
            model = model.to(device)

            # --- Функция потерь и оптимизатор (КЛЮЧЕВОЕ ИЗМЕНЕНИЕ) ---
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)# --- Цикл обучения ---
            logger.info(f"Начинаю обучение на {NUM_EPOCHS} эпох.")

            loss_log = []
            epoch_losses = []

            for epoch in range(NUM_EPOCHS):
                model.train()
                running_loss = 0.0
                batch_losses = []

                for i, batch_data in enumerate(train_loader):
                    if batch_data[0] is None:
                        continue

                    inputs, labels_0_indexed = batch_data
                    inputs = inputs.to(device)
                    ordinal_targets = label_to_ordinal_target(labels_0_indexed, NUM_CLASSES).to(device)

                    optimizer.zero_grad()
                    logits = model(inputs)
                    loss = criterion(logits, ordinal_targets)
                    loss.backward()
                    optimizer.step()

                    loss_value = loss.item()
                    running_loss += loss_value
                    batch_losses.append(loss_value)
                    loss_log.append(loss_value)  # для batch-графика, если хочешь

                    self.progress.emit(
                        f"Эпоха {epoch + 1}/{NUM_EPOCHS} | Батч {i + 1}/{len(train_loader)} | Loss: {loss_value:.4f}"
                    )

                epoch_avg_loss = running_loss / len(batch_losses) if batch_losses else 0
                epoch_losses.append(epoch_avg_loss)
                logger.info(f"Эпоха {epoch + 1} завершена. Средний loss: {epoch_avg_loss:.4f}")

            plt.plot(range(1, len(loss_log) + 1), loss_log)
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title('Batch\Loss')
            filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.jpg'
            plt.savefig(filename)
            plt.close()

            # --- Сохранение модели ---
            self.progress.emit("Сохраняю модель...")
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

            logger.success("Взвешенная Ordinal Regression модель успешно обучена и сохранена.")
            self.finished.emit(True, "Обучение завершено. Модель сохранена.")

        except Exception as e:
            logger.critical(f"Критическая ошибка в процессе обучения: {e}", exc_info=True)
            self.finished.emit(False, f"Ошибка обучения: {e}")