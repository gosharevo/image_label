# src/ml/model.py

import torch.nn as nn
from timm import create_model


class ViTOrdinal(nn.Module):
    """
    Каноническое определение модели Vision Transformer для порядковой регрессии.

    Эта модель использует backbone из библиотеки timm и заменяет его
    классификационную голову на линейный слой для порядковой регрессии,
    который имеет num_classes - 1 выходов.

    Этот класс является "единым источником истины" (Single Source of Truth)
    для архитектуры модели во всем проекте. Используйте его как для обучения,
    так и для инференса, чтобы гарантировать их полное соответствие.
    """

    def __init__(self, num_classes: int, pretrained: bool = False, drop_path_rate: float = 0.0):
        """
        Инициализация модели.

        Args:
            num_classes (int): Общее количество классов (например, 5 для шкалы от 1 до 5).

            pretrained (bool): Загружать ли предобученные на ImageNet веса для backbone.
                               - Установите True для старта обучения (transfer learning).
                               - Установите False для инференса, когда веса загружаются
                                 из вашего сохраненного файла (`state_dict`).

            drop_path_rate (float): Вероятность Stochastic Depth (DropPath). Это техника
                                    регуляризации, которая "выключает" целые блоки
                                    трансформера во время обучения. Должна быть одинаковой
                                    при обучении и инференсе для сохранения идентичности
                                    архитектуры.
        """
        super().__init__()

        # Создаем backbone. num_classes=0 удаляет исходную классификационную голову.
        # Это предпочтительный способ по сравнению с reset_classifier(0).
        self.backbone = create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate  # Передаем параметр в timm
        )

        # Динамически получаем количество входных признаков из backbone
        in_features = self.backbone.num_features

        # Определяем новую "голову" для порядковой регрессии
        self.head = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        """
        Прямой проход модели.

        Args:
            x (torch.Tensor): Входной тензор с изображениями формы (N, C, H, W).

        Returns:
            torch.Tensor: Логиты для порядковой регрессии формы (N, num_classes - 1).
        """
        # self.backbone(x) в timm для ViT по умолчанию возвращает признаки
        # CLS-токена после прохождения через все блоки трансформера.
        features = self.backbone(x)

        # Пропускаем признаки через нашу кастомную голову
        logits = self.head(features)

        return logits