import os
import hashlib
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel,
    QPushButton, QFileDialog, QSplitter, QGroupBox, QProgressBar, QMessageBox,
    QListWidgetItem, QStyle, QDialog, QMenuBar, QMenu, QAction
)
from PyQt5.QtGui import QPixmap, QKeySequence, QColor, QIcon, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent

from loguru import logger
from pathlib import Path

# --- LOCAL IMPORTS ---
from src.app.state_manager import StateManager
from src.utils.config import (
    WINDOW_TITLE, INFERENCE_BATCH_SIZE, MODEL_VERDICT_HEADER,
    MODEL_PREDICTION_TEXT, MODEL_ALREADY_LABELED_TEXT,
    MODEL_NOT_TRAINED_TEXT, MODEL_NO_PREDICTION_TEXT, DEFAULT_START_FOLDER,
    MODEL_PATH, GRAD_CAM_DIR, NUM_CLASSES, BATCH_SIZE
)
from src.ml.trainer import Trainer
from src.ml.predictor import Predictor

# --- APPLE DESIGN STYLESHEET (без изменений) ---
STYLESHEET = """
QWidget {
    background-color: #1e1e1e;
    color: #f0f0f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10pt;
}
QMainWindow, QDialog {
    background-color: #1e1e1e;
}
QMenuBar {
    background-color: #2d2d2d;
    border-bottom: 1px solid #3c3c3c;
    padding: 4px;
}
QMenuBar::item {
    padding: 6px 12px;
    border-radius: 6px;
}
QMenuBar::item:selected, QMenuBar::item:pressed {
    background-color: #007aff;
    color: white;
}
QMenuBar::item:disabled {
    color: #5a5a5a;
}
QMenu {
    background-color: #2d2d2d;
    border: 1px solid #3c3c3c;
    border-radius: 8px;
    padding: 5px;
}
QMenu::item {
    padding: 8px 24px 8px 12px;
    border-radius: 6px;
}
QMenu::item:selected {
    background-color: #007aff;
    color: white;
}
QMenu::separator {
    height: 1px;
    background: #3c3c3c;
    margin: 5px 0;
}
QListWidget {
    background-color: #2d2d2d;
    border: none;
    border-radius: 8px;
    padding: 5px;
}
QListWidget::item {
    padding: 10px;
    border-radius: 6px;
}
QListWidget::item:hover:!selected {
    background-color: #3a3a3a;
}
QListWidget::item:selected {
    background-color: #007aff;
    color: #ffffff;
}
QPushButton {
    background-color: #007aff;
    color: white;
    border: none;
    padding: 10px 16px;
    border-radius: 8px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #005ecb;
}
QPushButton:pressed {
    background-color: #004bad;
}
QPushButton:disabled {
    background-color: #4a4a4a;
    color: #8e8e8e;
}
QGroupBox {
    background-color: #2d2d2d;
    border: none;
    border-radius: 8px;
    margin-top: 1em;
    padding: 15px;
}
QGroupBox::title {
    color: #f0f0f0;
    font-size: 11pt;
    font-weight: 600;
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    top: 12px;
}
QLabel#image_label {
    background-color: #000000;
    border: 1px solid #3c3c3c;
    border-radius: 8px;
}
QLabel#grad_cam_label {
    background-color: #252525;
    border: 1px dashed #4a4a4a;
    border-radius: 6px;
    color: #8e8e8e;
}
QProgressBar {
    height: 8px;
    border: none;
    border-radius: 4px;
    background-color: #3c3c3c;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #007aff;
    border-radius: 4px;
}
QSplitter::handle {
    background-color: #3c3c3c;
    width: 1px;
}
QSplitter::handle:horizontal {
    margin: 0 4px;
}
QSplitter::handle:vertical {
    margin: 4px 0;
}
QSplitter::handle:hover {
    background-color: #007aff;
}
QScrollBar:vertical, QScrollBar:horizontal {
    border: none;
    background: #1e1e1e;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #4a4a4a;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: #5a5a5a;
}
QScrollBar::add-line, QScrollBar::sub-line {
    border: none;
    background: none;
    height: 0px;
    width: 0px;
}
"""


class MainWindow(QMainWindow):
    request_ui_update = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.state = StateManager()
        self.current_image_path = None
        self.original_pixmap: QPixmap = None

        self.is_training = False
        self.is_inferring = False

        self._is_updating_image = False
        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self._delayed_ui_rescale)

        self.thread = None
        self.worker = None

        self._init_ui()
        self.request_ui_update.connect(self.update_ui_elements)
        logger.info("UI инициализировано c пакетом 'Presidential Suite'.")

    def is_busy(self) -> bool:
        return self.is_training or self.is_inferring

    def _init_ui(self):
        self.setWindowTitle(WINDOW_TITLE)
        self.setStyleSheet(STYLESHEET)

        app_icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        self.setWindowIcon(app_icon)

        self._create_menu_bar()

        central_container = QWidget()
        main_layout = QHBoxLayout(central_container)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        main_splitter = QSplitter(Qt.Horizontal)

        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)
        center_panel = self._create_center_panel()
        main_splitter.addWidget(center_panel)
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)

        main_splitter.setSizes([250, 700, 300])
        main_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(main_splitter)
        self.setCentralWidget(central_container)
        self._setup_status_bar()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")

        open_icon = self.style().standardIcon(QStyle.SP_DirOpenIcon)
        self.open_action = QAction(QIcon(open_icon), "Выбрать папку...", self)
        self.open_action.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(self.open_action)

    def _create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        list_label = QLabel("Изображения")
        list_label.setFont(QFont(self.font().family(), 11, QFont.Bold))
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentItemChanged.connect(self.on_file_selected)
        left_layout.addWidget(list_label)
        left_layout.addWidget(self.file_list_widget)
        return left_panel

    def _create_center_panel(self):
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(12)
        self.image_label = QLabel("Выберите папку с изображениями\n(Файл → Выбрать папку...)")
        self.image_label.setObjectName("image_label")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        # --- НОВАЯ ФУНКЦИЯ: Включаем отслеживание мыши для контекстного меню ---
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.show_image_context_menu)

        rating_box = QGroupBox(f"Оценка (Клавиши 1-{NUM_CLASSES if NUM_CLASSES < 10 else 9})")
        rating_layout = QHBoxLayout()
        rating_layout.setSpacing(8)
        self.rating_buttons = []
        for i in range(1, NUM_CLASSES + 1):
            btn = QPushButton(str(i))
            btn.setFixedWidth(40)
            btn.setToolTip(f"Присвоить класс {i}")
            if i < 10:
                btn.setShortcut(QKeySequence(str(i)))
            btn.clicked.connect(lambda _, lbl=i: self.rate_image(lbl))

            # --- НОВАЯ ФУНКЦИЯ: Устанавливаем фильтр событий для подсказок ---
            btn.installEventFilter(self)

            rating_layout.addWidget(btn)
            self.rating_buttons.append(btn)
        rating_box.setLayout(rating_layout)

        center_layout.addWidget(self.image_label, 1)
        center_layout.addWidget(rating_box)
        return center_panel

    def _create_right_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        prediction_box = QGroupBox(MODEL_VERDICT_HEADER)
        prediction_layout = QVBoxLayout(prediction_box)
        self.prediction_label = QLabel(MODEL_NOT_TRAINED_TEXT)
        self.prediction_label.setWordWrap(True)
        self.prediction_label.setStyleSheet("color: #a0a0a0; font-size: 9pt;")
        prediction_layout.addWidget(self.prediction_label)

        control_box = QGroupBox("Управление моделью")
        control_layout = QVBoxLayout(control_box)
        control_layout.setSpacing(10)
        self.train_button = QPushButton("Обучить модель")
        self.train_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.infer_button = QPushButton("Получить подсказки")
        self.infer_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.train_button.clicked.connect(self.start_training)
        self.infer_button.clicked.connect(self.start_inference)
        control_layout.addWidget(self.train_button)
        control_layout.addWidget(self.infer_button)

        self.analysis_box = QGroupBox("Анализ")
        analysis_layout = QVBoxLayout(self.analysis_box)
        self.grad_cam_label = QLabel("Нет данных для анализа")
        self.grad_cam_label.setObjectName("grad_cam_label")
        self.grad_cam_label.setAlignment(Qt.AlignCenter)
        self.grad_cam_label.setMinimumHeight(150)
        # --- НОВАЯ ФУНКЦИЯ: Контекстное меню для Grad-CAM ---
        self.grad_cam_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.grad_cam_label.customContextMenuRequested.connect(self.show_image_context_menu)

        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: stretch-фактор применяется к QLabel внутри компоновки ---
        analysis_layout.addWidget(self.grad_cam_label, 1)  # '1' заставляет QLabel растягиваться
        self.analysis_box.setVisible(False)

        right_layout.addWidget(prediction_box)
        right_layout.addWidget(control_box)
        right_layout.addWidget(self.analysis_box, 1)

        return right_panel

    def _setup_status_bar(self):
        self.statusBar = self.statusBar()
        self.statusBar.setStyleSheet("padding: 3px;")

        # --- НОВАЯ ФУНКЦИЯ: Интерактивный счетчик ---
        self.annotation_counter_label = QLabel("")
        self.annotation_counter_label.setStyleSheet("padding: 0 5px; color: #a0a0a0;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)

        self.statusBar.addPermanentWidget(self.annotation_counter_label, 0)
        self.statusBar.addPermanentWidget(self.progress_bar, 1)
        self.statusBar.showMessage("Готов")

    # --- НОВАЯ ФУНКЦИЯ: Обработка событий для динамических подсказок ---
    def eventFilter(self, source, event):
        if event.type() == QEvent.Enter and source in self.rating_buttons:
            self.statusBar.showMessage(f"Присвоить класс: {source.text()}", 2000)
        elif event.type() == QEvent.Leave and source in self.rating_buttons:
            self.statusBar.clearMessage()
        return super().eventFilter(source, event)

    # --- НОВАЯ ФУНКЦИЯ: Навигация и полноэкранный режим ---
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Right:
            current_row = self.file_list_widget.currentRow()
            if current_row < self.file_list_widget.count() - 1:
                self.file_list_widget.setCurrentRow(current_row + 1)
        elif key == Qt.Key_Left:
            current_row = self.file_list_widget.currentRow()
            if current_row > 0:
                self.file_list_widget.setCurrentRow(current_row - 1)
        elif key in (Qt.Key_F, Qt.Key_F11):
            self.toggle_fullscreen()
        elif key == Qt.Key_Escape and self.isFullScreen():
            self.toggle_fullscreen(force_exit=True)
        else:
            super().keyPressEvent(event)

    def toggle_fullscreen(self, force_exit=False):
        if self.isFullScreen() or force_exit:
            self.showMaximized()
        elif self.original_pixmap:
            self.showFullScreen()

    def mouseDoubleClickEvent(self, event):
        # Позволяет выйти из полноэкранного режима двойным кликом
        if self.isFullScreen():
            self.toggle_fullscreen(force_exit=True)
        super().mouseDoubleClickEvent(event)

    # --- НОВАЯ ФУНКЦИЯ: Контекстное меню ---
    def show_image_context_menu(self, position):
        if not self.current_image_path:
            return

        menu = QMenu()
        menu.setStyleSheet(STYLESHEET)  # Применяем наш дизайн
        copy_path_action = menu.addAction("Копировать путь к файлу")
        action = menu.exec_(self.image_label.mapToGlobal(position))

        if action == copy_path_action:
            QApplication.clipboard().setText(self.current_image_path)
            self.statusBar.showMessage("Путь скопирован в буфер обмена", 2000)

    def _update_ui_for_task_state(self):
        busy = self.is_busy()
        self.train_button.setEnabled(not busy)
        self.infer_button.setEnabled(not busy)
        self.open_action.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        if not busy:
            self.statusBar.showMessage("Готов", 2000)
            self.progress_bar.setValue(0)
        self.update_grad_cam_display()

    # --- НОВАЯ ФУНКЦИЯ: Обновление счетчика ---
    def _update_annotation_counter(self):
        annotated_count = len(self.state.annotations)
        total_count = len(self.state.all_image_paths)
        if total_count > 0:
            self.annotation_counter_label.setText(f"Размечено: {annotated_count} / {total_count}")
        else:
            self.annotation_counter_label.setText("")

    def show(self):
        super().showMaximized()

    def on_file_selected(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        if current_item:
            self.current_image_path = current_item.data(Qt.UserRole)
            self.original_pixmap = QPixmap(self.current_image_path)
            if self.original_pixmap.isNull():
                self.original_pixmap = None
        else:
            self.current_image_path = None
            self.original_pixmap = None
        self.update_image_display()
        self.update_prediction_panel()
        self.update_grad_cam_display()

    def update_image_display(self):
        # В полноэкранном режиме центральный виджет скрыт, но мы можем его обновить
        target_label = self.centralWidget().findChild(QLabel, "image_label") if not self.isFullScreen() else self
        pixmap = self.original_pixmap

        if self._is_updating_image or not target_label:
            return
        self._is_updating_image = True
        try:
            if pixmap and not pixmap.isNull():
                scaled = pixmap.scaled(target_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                target_label.setPixmap(scaled)
            else:
                target_label.setPixmap(QPixmap())
                if isinstance(target_label, QLabel):
                    message = "Изображение не найдено" if self.current_image_path else "Изображение не выбрано"
                    target_label.setText(message)
        finally:
            self._is_updating_image = False

    def _delayed_ui_rescale(self):
        self.update_image_display()
        self.update_grad_cam_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_timer.start(50)
        if self.isFullScreen():  # Обновляем картинку при изменении размера экрана
            self.update_image_display()

    def open_folder_dialog(self):
        if self.is_busy(): return
        folder_path = QFileDialog.getExistingDirectory(self, "Выберите папку", DEFAULT_START_FOLDER)
        if folder_path:
            self.state.load_images_from_folder(folder_path)
            self.populate_file_list()

    def populate_file_list(self):
        current_path_before_reload = self.current_image_path
        self.file_list_widget.clear()
        annotated_color = QColor("#8e8e8e")
        item_to_select = None
        for path in self.state.all_image_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            if path in self.state.annotations:
                item.setForeground(annotated_color)
            self.file_list_widget.addItem(item)
            if path == current_path_before_reload:
                item_to_select = item

        self._update_annotation_counter()

        if item_to_select:
            self.file_list_widget.setCurrentItem(item_to_select)
        elif self.file_list_widget.count() > 0:
            self.file_list_widget.setCurrentRow(0)

    def rate_image(self, label: int):
        if not self.current_image_path: return
        current_row = self.file_list_widget.currentRow()
        if self.state.update_annotation(self.current_image_path, label):
            item = self.file_list_widget.item(current_row)
            if item:
                item.setForeground(QColor("#8e8e8e"))

            self._update_annotation_counter()  # Обновляем счетчик
            self.update_prediction_panel()

            if current_row + 1 < self.file_list_widget.count():
                self.file_list_widget.setCurrentRow(current_row + 1)
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить аннотацию. Подробности в логе.")

    def update_ui_elements(self):
        self.update_image_display()
        self.update_prediction_panel()
        self.update_grad_cam_display()
        self._update_ui_for_task_state()
        self._update_annotation_counter()

    def update_prediction_panel(self):
        if not self.current_image_path:
            self.prediction_label.setText("")
            return
        if self.current_image_path in self.state.annotations:
            class_name = self.state.annotations.get(self.current_image_path, "")
            self.prediction_label.setText(f"{MODEL_ALREADY_LABELED_TEXT} {class_name}")
        elif not os.path.exists(MODEL_PATH):
            self.prediction_label.setText(MODEL_NOT_TRAINED_TEXT)
        elif self.current_image_path in self.state.predictions:
            pred_class, prob = self.state.predictions[self.current_image_path]
            self.prediction_label.setText(MODEL_PREDICTION_TEXT.format(pred_class, prob))
        else:
            self.prediction_label.setText(MODEL_NO_PREDICTION_TEXT)
        self.update_grad_cam_display()

    def get_grad_cam_path(self) -> Path | None:
        if not self.current_image_path:
            return None
        grad_cam_dir_path = Path(GRAD_CAM_DIR)
        path_hash = hashlib.sha256(self.current_image_path.encode()).hexdigest()
        return grad_cam_dir_path / f"{path_hash}.jpg"

    def update_grad_cam_display(self):
        path = self.get_grad_cam_path()
        if path and path.exists() and not self.is_busy():
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                self.analysis_box.setVisible(True)
                scaled_pixmap = pixmap.scaled(self.grad_cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.grad_cam_label.setPixmap(scaled_pixmap)
            else:
                self.analysis_box.setVisible(True)
                self.grad_cam_label.setText("Ошибка\nзагрузки")
        else:
            self.analysis_box.setVisible(False)

    def start_training(self):
        if self.is_busy(): return
        dataset = self.state.get_full_dataset_for_training()
        if len(dataset) < BATCH_SIZE:
            QMessageBox.warning(self, "Недостаточно данных", f"Нужно разметить минимум {BATCH_SIZE} изображений.")
            return

        self.is_training = True
        self._update_ui_for_task_state()
        self.statusBar.showMessage("Идет обучение...")

        self.thread = QThread()
        self.worker = Trainer(dataset)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.progress.connect(self.statusBar.showMessage)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_training_finished(self, success: bool, message: str):
        self.is_training = False
        self._update_ui_for_task_state()
        if success:
            QMessageBox.information(self, "Успех", message)
            self.statusBar.showMessage(message, 5000)
        else:
            QMessageBox.critical(self, "Ошибка обучения", message)
        self.request_ui_update.emit()

    def start_inference(self):
        if self.is_busy(): return
        if not os.path.exists(MODEL_PATH):
            QMessageBox.warning(self, "Модель не найдена", "Сначала необходимо обучить модель.")
            return

        unannotated = self.state.get_unannotated_images()
        to_predict = [p for p in unannotated if p not in self.state.predictions][:INFERENCE_BATCH_SIZE]
        if not to_predict:
            QMessageBox.information(self, "Все размечено", "Нет новых изображений для анализа.")
            return

        self.is_inferring = True
        self._update_ui_for_task_state()
        self.statusBar.showMessage(f"Анализ {len(to_predict)} изображений...")

        self.thread = QThread()
        self.worker = Predictor(to_predict)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.progress.connect(self.statusBar.showMessage)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_inference_finished(self, predictions: dict):
        self.is_inferring = False
        self._update_ui_for_task_state()
        if predictions:
            self.state.update_predictions(predictions)
            QMessageBox.information(self, "Готово", f"Получено {len(predictions)} новых подсказок.")
        else:
            QMessageBox.warning(self, "Нет результата", "Не удалось получить предсказания. См. логи.")
        self.request_ui_update.emit()

    def closeEvent(self, event):
        if self.is_busy():
            reply = QMessageBox.question(self, 'Подтверждение',
                                         "Фоновый процесс еще выполняется. Вы уверены, что хотите выйти?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                if self.thread and self.thread.isRunning():
                    logger.warning("Попытка корректно завершить поток...")
                    self.thread.quit()
                    if not self.thread.wait(2000):
                        logger.error("Поток не отвечает. Принудительное завершение.")
                        self.thread.terminate()
                        self.thread.wait(500)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

