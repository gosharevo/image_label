import sys
from PyQt5.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from src.utils.logger import setup_logger, logger


def main():
    # 1. Инициализация логгера. Это первый и самый важный шаг.
    setup_logger()
    logger.info("Запуск приложения.")

    # 2. Создание QApplication
    app = QApplication(sys.argv)

    # 3. Создание и отображение главного окна
    try:
        window = MainWindow()
        window.show()
    except Exception as e:
        logger.critical(f"Критическая ошибка при инициализации главного окна: {e}")
        logger.exception(e)  # Логируем полный traceback
        sys.exit(1)  # Завершаем приложение с кодом ошибки

    # 4. Запуск основного цикла событий
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
