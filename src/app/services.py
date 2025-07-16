# src/app/services.py

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from typing import List, Dict, Any
from loguru import logger

from src.ml.trainer import Trainer
from src.ml.predictor import Predictor, PredictionResult


class MLService(QObject):
    """
    Сервисный слой, который инкапсулирует всю логику взаимодействия с ML-моделями.
    UI-слой общается только с этим классом.
    """
    training_started = pyqtSignal()
    training_finished = pyqtSignal(bool, str)
    training_progress = pyqtSignal(str)

    inference_started = pyqtSignal()
    inference_finished = pyqtSignal(dict)
    inference_progress = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None

    def start_training(self, dataset: List[tuple]):
        logger.info("Сервис получил запрос на обучение.")
        self.thread = QThread()
        self.worker = Trainer(dataset)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.training_progress)
        self.worker.finished.connect(self._on_task_finished)
        self.worker.finished.connect(self.training_finished)

        self.thread.start()
        self.training_started.emit()

    def start_inference(self, image_paths: List[str]):
        logger.info(f"Сервис получил запрос на инференс для {len(image_paths)} изображений.")
        self.thread = QThread()
        self.worker = Predictor()  # Predictor теперь не принимает пути в конструкторе

        # Передаем параметры в метод run
        self.thread.started.connect(lambda: self.worker.run(image_paths))
        self.worker.progress.connect(self.inference_progress)
        self.worker.finished.connect(self._on_task_finished)
        self.worker.finished.connect(self.inference_finished)

        self.thread.start()
        self.inference_started.emit()

    def _on_task_finished(self):
        """Очищает ресурсы после завершения задачи."""
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None
        logger.debug("Поток и воркер ML-задачи успешно очищены.")