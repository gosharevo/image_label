# src/app/main_window.py

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QLabel,
    QPushButton, QFileDialog, QSplitter, QGroupBox, QProgressBar,
    QMessageBox, QListWidgetItem, QStyle, QAction, QLineEdit, QComboBox,
    QMenu, QApplication
)
from PyQt5.QtGui import QPixmap, QKeySequence, QColor, QIcon, QFont, QBrush
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize
import os
import shutil
from pathlib import Path
from loguru import logger

# Компоненты приложения
from src.app.state_manager import StateManager, AppState
from src.app.services import MLService
from src.utils.config import (
    WINDOW_TITLE, INFERENCE_BATCH_SIZE, MODEL_VERDICT_HEADER,
    MODEL_PREDICTION_TEXT, MODEL_ALREADY_LABELED_TEXT,
    MODEL_NOT_TRAINED_TEXT, MODEL_NO_PREDICTION_TEXT,
    DEFAULT_START_FOLDER, MODEL_PATH, BATCH_SIZE
)
from src.ml.predictor import PredictionResult

try:
    with open(Path(__file__).parent.parent / 'ui' / 'style.css', 'r') as f:
        STYLESHEET = f.read()
except FileNotFoundError:
    logger.warning("Файл стилей style.css не найден. Будет использован стандартный вид.")
    STYLESHEET = ""


# --- Специализированные виджеты ---

class FileListPanel(QWidget):
    """Панель со списком файлов, поиском и фильтрацией."""
    file_selected = pyqtSignal(str)

    def __init__(self, state: StateManager, parent=None):
        super().__init__(parent)
        self.state = state
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self);
        layout.setContentsMargins(0, 0, 0, 0);
        layout.setSpacing(10)
        lbl = QLabel("Изображения");
        lbl.setFont(QFont(self.font().family(), 11, QFont.Bold))

        self.search_box = QLineEdit();
        self.search_box.setPlaceholderText("Поиск по имени...")
        self.filter_box = QComboBox();
        self.filter_box.addItem("Все метки", None)
        for i in range(1, self.state.config.num_classes + 1): self.filter_box.addItem(f"Метка {i}", i)

        self.file_list_widget = QListWidget()

        self.search_box.textChanged.connect(self.populate)
        self.filter_box.currentIndexChanged.connect(self.populate)
        self.file_list_widget.currentItemChanged.connect(self._on_selection_changed)

        layout.addWidget(lbl);
        layout.addWidget(self.search_box)
        layout.addWidget(self.filter_box);
        layout.addWidget(self.file_list_widget)

    def _on_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        if current: self.file_selected.emit(current.data(Qt.UserRole))

    def populate(self):
        current_path = self.state.current_image_path
        self.file_list_widget.clear()

        query = self.search_box.text().lower()
        selected_label = self.filter_box.currentData()

        annotated_color = QBrush(QColor("#808080"))  # Серый для размеченных

        item_to_select = None
        for path in self.state.all_image_paths:
            name = Path(path).name
            # if query and query not in name.lower(): continue
            if query and query not in str(path).lower(): continue

            annotation = self.state.annotations.get(path)
            if selected_label is not None and annotation != selected_label: continue

            item = QListWidgetItem(name);
            item.setData(Qt.UserRole, path)
            if annotation:
                item.setForeground(annotated_color)

            self.file_list_widget.addItem(item)
            if path == current_path: item_to_select = item

        if item_to_select: self.file_list_widget.setCurrentItem(item_to_select)

    def select_first_unannotated(self):
        """Ищет и выбирает первый неразмеченный элемент."""
        for i in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(i)
            path = item.data(Qt.UserRole)
            if path not in self.state.annotations:
                self.file_list_widget.setCurrentItem(item)
                return
        if self.file_list_widget.count() > 0:
            self.file_list_widget.setCurrentRow(0)

    def select_next_item(self):
        """Просто выбирает следующий элемент в списке."""
        current_row = self.file_list_widget.currentRow()
        if current_row + 1 < self.file_list_widget.count():
            self.file_list_widget.setCurrentRow(current_row + 1)


class ImageViewerPanel(QWidget):
    """Панель с изображением и кнопками для оценки."""
    image_rated = pyqtSignal(int)

    def __init__(self, state: StateManager, parent=None):
        super().__init__(parent)
        self.state = state
        self._init_ui()
        self.original_pixmap = None

    def _init_ui(self):
        layout = QVBoxLayout(self);
        layout.setContentsMargins(0, 0, 0, 0);
        layout.setSpacing(12)
        self.image_label = QLabel("Выберите папку (Файл → Выбрать папку...)")
        self.image_label.setObjectName("image_label")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)

        rating_box = QGroupBox(f"Оценка (Клавиши 1-{self.state.config.num_classes})")
        rating_layout = QHBoxLayout();
        rating_box.setLayout(rating_layout)
        for i in range(1, self.state.config.num_classes + 1):
            btn = QPushButton(str(i));
            btn.setFixedWidth(40)
            btn.setShortcut(QKeySequence(str(i)))
            btn.clicked.connect(lambda _, lbl=i: self.image_rated.emit(lbl))
            rating_layout.addWidget(btn)

        layout.addWidget(self.image_label, 1);
        layout.addWidget(rating_box)

    def update_display(self):
        path = self.state.current_image_path
        if not path:
            self.image_label.clear();
            self.original_pixmap = None
            return

        self.original_pixmap = QPixmap(path)
        if self.original_pixmap.isNull():
            self.image_label.setText("Ошибка загрузки изображения")
            self.original_pixmap = None
        else:
            self.rescale_pixmap()

    def rescale_pixmap(self):
        if self.original_pixmap and self.image_label.width() > 10:
            scaled = self.original_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)


class ControlPanel(QWidget):
    """Панель с кнопками управления ML, предсказаниями и Grad-CAM."""
    train_requested = pyqtSignal()
    inference_requested = pyqtSignal()

    def __init__(self, state: StateManager, parent=None):
        super().__init__(parent)
        self.state = state
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self);
        layout.setContentsMargins(0, 0, 0, 0);
        layout.setSpacing(12)

        pred_group = QGroupBox(MODEL_VERDICT_HEADER);
        pred_layout = QVBoxLayout(pred_group)
        self.prediction_label = QLabel(MODEL_NOT_TRAINED_TEXT);
        self.prediction_label.setWordWrap(True)
        pred_layout.addWidget(self.prediction_label)

        ctrl_group = QGroupBox("Управление моделью");
        ctrl_layout = QVBoxLayout(ctrl_group)
        self.train_button = QPushButton("Обучить модель");
        self.train_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOkButton))
        self.infer_button = QPushButton("Получить подсказки");
        self.infer_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        ctrl_layout.addWidget(self.train_button);
        ctrl_layout.addWidget(self.infer_button)
        self.train_button.clicked.connect(self.train_requested)
        self.infer_button.clicked.connect(self.inference_requested)

        self.analysis_box = QGroupBox("Анализ модели")
        analysis_layout = QVBoxLayout(self.analysis_box)
        self.viz_label = QLabel("Нет данных для анализа")
        self.viz_label.setObjectName("viz_label");
        self.viz_label.setAlignment(Qt.AlignCenter)
        self.viz_label.setMinimumHeight(150)
        analysis_layout.addWidget(self.viz_label, 1)
        self.analysis_box.setVisible(False)

        layout.addWidget(pred_group);
        layout.addWidget(ctrl_group);
        layout.addWidget(self.analysis_box, 1)

    def update_panel(self):
        self._update_prediction_text()
        self._update_visualization()
        self._update_button_state()

    def _update_prediction_text(self):
        path = self.state.current_image_path
        if not path: self.prediction_label.setText(""); return

        annotation = self.state.annotations.get(path)
        prediction = self.state.predictions.get(path)

        if annotation:
            text = MODEL_ALREADY_LABELED_TEXT.format(annotation)
        elif prediction:
            text = MODEL_PREDICTION_TEXT.format(prediction.predicted_class, prediction.confidence)
        elif not Path(MODEL_PATH).exists():
            text = MODEL_NOT_TRAINED_TEXT
        else:
            text = MODEL_NO_PREDICTION_TEXT
        self.prediction_label.setText(text)

    def _update_visualization(self):
        prediction = self.state.predictions.get(self.state.current_image_path)
        if prediction and prediction.visualization_path and Path(prediction.visualization_path).exists():
            pixmap = QPixmap(str(prediction.visualization_path))
            if not pixmap.isNull():
                self.analysis_box.setVisible(True)
                scaled = pixmap.scaled(self.viz_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.viz_label.setPixmap(scaled)
                return
        self.analysis_box.setVisible(False)
        self.viz_label.clear()

    def _update_button_state(self):
        is_busy = self.state.app_state != AppState.IDLE
        self.train_button.setEnabled(not is_busy)
        self.infer_button.setEnabled(not is_busy)


# --- Основное окно приложения ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = StateManager()
        self.ml_service = MLService()
        self.resize_timer = QTimer(self);
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self._on_delayed_resize)
        self._init_ui()
        self._connect_signals()

        # ЗАПУСК В ПОЛНОЭКРАННОМ РЕЖИМЕ
        # self.showFullScreen()

        logger.info("Приложение успешно инициализировано в полноэкранном режиме.")

    def _init_ui(self):
        self.setWindowTitle(WINDOW_TITLE)
        self.setStyleSheet(STYLESHEET)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))

        self._create_menu_bar()
        self._setup_status_bar()

        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)

        self.file_list_panel = FileListPanel(self.state)
        self.image_viewer_panel = ImageViewerPanel(self.state)
        self.control_panel = ControlPanel(self.state)

        splitter.addWidget(self.file_list_panel)
        splitter.addWidget(self.image_viewer_panel)
        splitter.addWidget(self.control_panel)
        splitter.setSizes([250, 700, 300]);
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        self.open_action = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), "Выбрать папку...", self)
        file_menu.addAction(self.open_action)

        view_menu = menu_bar.addMenu("Вид")
        self.fullscreen_action = QAction("Полноэкранный режим (F11)", self, checkable=True)
        self.fullscreen_action.setShortcut(QKeySequence(Qt.Key_F11))
        self.fullscreen_action.setChecked(self.isFullScreen())
        self.fullscreen_action.toggled.connect(self.toggle_fullscreen)
        view_menu.addAction(self.fullscreen_action)

    def _setup_status_bar(self):
        self.status_bar = self.statusBar()
        self.annotation_counter_label = QLabel("")
        self.progress_bar = QProgressBar();
        self.progress_bar.setVisible(False);
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.annotation_counter_label, 0)
        self.status_bar.addPermanentWidget(self.progress_bar, 1)
        self.status_bar.showMessage("Готов")

    def _connect_signals(self):
        self.open_action.triggered.connect(self._on_open_folder_requested)
        self.file_list_panel.file_selected.connect(self._on_file_selected)
        self.image_viewer_panel.image_rated.connect(self._on_image_rated)
        self.control_panel.train_requested.connect(self._on_train_requested)
        self.control_panel.inference_requested.connect(self._on_inference_requested)

        self.ml_service.training_started.connect(lambda: self.state.set_app_state(AppState.TRAINING))
        self.ml_service.inference_started.connect(lambda: self.state.set_app_state(AppState.INFERRING))
        self.ml_service.training_progress.connect(self.status_bar.showMessage)
        self.ml_service.inference_progress.connect(self.status_bar.showMessage)
        self.ml_service.training_finished.connect(self._on_training_finished)
        self.ml_service.inference_finished.connect(self._on_inference_finished)

        self.state.state_changed.connect(self._on_state_changed)
        self.state.file_list_updated.connect(self._on_file_list_updated)
        self.state.annotations_changed.connect(self._update_annotation_counter)

    def _on_file_list_updated(self):
        self.file_list_panel.populate()
        self.file_list_panel.select_first_unannotated()

    def _on_open_folder_requested(self):
        if self.state.app_state != AppState.IDLE: return
        folder_path = QFileDialog.getExistingDirectory(self, "Выберите папку", DEFAULT_START_FOLDER)
        if folder_path: self.state.load_images_from_folder(folder_path)

    def _on_file_selected(self, path: str):
        self.state.set_current_image(path)

    def _on_image_rated(self, label: int):
        if not self.state.current_image_path: return
        self.state.update_annotation(self.state.current_image_path, label)
        self.file_list_panel.populate()
        self.file_list_panel.select_next_item()

    def _on_train_requested(self):
        dataset = self.state.get_full_dataset_for_training()
        if len(dataset) < BATCH_SIZE:
            QMessageBox.warning(self, "Недостаточно данных", f"Нужно разметить минимум {BATCH_SIZE} изображений.")
            return
        self.ml_service.start_training(dataset)

    def _on_inference_requested(self):
        if not Path(MODEL_PATH).exists():
            QMessageBox.warning(self, "Модель не найдена", "Сначала необходимо обучить модель.")
            return
        unannotated = self.state.get_unannotated_images()
        to_predict = [p for p in unannotated if p not in self.state.predictions][:INFERENCE_BATCH_SIZE]
        if not to_predict:
            QMessageBox.information(self, "Все размечено", "Нет новых изображений для анализа.")
            return
        self.ml_service.start_inference(to_predict)

    def _on_training_finished(self, success: bool, message: str):
        self.state.set_app_state(AppState.IDLE)
        if success:
            QMessageBox.information(self, "Успех", message)
        else:
            QMessageBox.critical(self, "Ошибка обучения", message)

    def _on_inference_finished(self, predictions: dict):
        self.state.set_app_state(AppState.IDLE)
        if predictions:
            self.state.update_predictions(predictions)
            QMessageBox.information(self, "Готово", f"Получено {len(predictions)} новых подсказок.")
            self.control_panel.update_panel()
        else:
            QMessageBox.warning(self, "Нет результата", "Не удалось получить предсказания. См. логи.")

    def _on_state_changed(self, new_state: AppState):
        is_busy = new_state != AppState.IDLE
        self.open_action.setEnabled(not is_busy)
        self.progress_bar.setVisible(is_busy)
        if not is_busy: self.status_bar.showMessage("Готов", 2000)

        self.image_viewer_panel.update_display()
        self.control_panel.update_panel()

    def _update_annotation_counter(self):
        annotated = len(self.state.annotations)
        total = len(self.state.all_image_paths)
        if total:
            self.annotation_counter_label.setText(f"Размечено: {annotated}/{total}")
        else:
            self.annotation_counter_label.setText("")

    def _on_delayed_resize(self):
        self.image_viewer_panel.rescale_pixmap()
        self.control_panel._update_visualization()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fullscreen_action: self.fullscreen_action.setChecked(self.isFullScreen())
        self.resize_timer.start(50)

    def closeEvent(self, event):
        if self.state.app_state != AppState.IDLE:
            reply = QMessageBox.question(self, "Подтверждение", "Фоновая задача еще выполняется. Выйти?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                event.ignore();
                return

        self.state.backup_annotations()
        event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Right:
            self.file_list_panel.select_next_item()
        elif key == Qt.Key_Left:
            current_row = self.file_list_panel.file_list_widget.currentRow()
            if current_row > 0:
                self.file_list_panel.file_list_widget.setCurrentRow(current_row - 1)
        elif key == Qt.Key_Escape and self.isFullScreen():
            self.toggle_fullscreen(False)
        else:
            super().keyPressEvent(event)

    def toggle_fullscreen(self, checked):
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()
