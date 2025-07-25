STYLE_SHEET = """
            QMainWindow {
                background-color: #f5f5f5;
            }

            QMenuBar {
                background-color: #ffffff;
                color: #000;
                padding-bottom: 5px;  /* Increase padding to the bottom */
                border-bottom: 1px solid #cccccc;
            }

            QMenuBar::item {
                background-color: #ffffff;
                color: #000;
                padding: 6px 10px;  /* Increase padding to create more space */
                margin: 0.5;
            }

            QMenuBar::item:selected {
                background-color: #e0e0e0;
            }

            QMenu {
                background-color: #ffffff;
                color: #000;
                margin: 0;
            }

            QMenu::item {
                background-color: #ffffff;
                color: #000;
                padding: 4px 10px;
            }

            QMenu::item:selected {
                background-color: #e0e0e0;
            }

            QToolBar {
                background-color: #ffffff;
                padding: 5px 0;  /* Add padding to the top and bottom */
                margin: 0;
                border-top: 1px solid #cccccc;  /* Add top border to create separation */
                border-bottom: 1px solid #cccccc;  /* Add bottom border to create separation */
            }

            QToolButton {
                background-color: #ffffff;
                border: none;
                color: #000;
                padding: 5px;
                margin: 0.5px;
            }

            QToolButton:hover {
                background-color: #e0e0e0;
            }

            QSplitter::handle {
                background-color: #dcdcdc;
            }

            QTabWidget::pane {
                border: 1px solid #cccccc;
                background: #f5f5f5;
            }

            QTabBar::tab {
                background: #ffffff;
                border: 1px solid #cccccc;
                padding: 5px;
                margin: 2px;
            }

            QTabBar::tab:selected {
                background: #e0e0e0;
                border-bottom-color: #f5f5f5;
            }
            
            /* QTreeView */

            QTreeView {
                background-color: #ffffff;
                color: #000;
                border: 1px solid #cccccc;
                padding: 5px;
                show-decoration-selected: 0;
            }
            
            QTreeView::item:selected {
                color: #000;
                background-color: #3294f0;
                border-radius: 2px;
            }

            QTextEdit, QPlainTextEdit {
                background-color: #ffffff;
                color: #000;
                border: 1px solid #cccccc;
                padding: 5px;
            }

            QPushButton {
                background-color: #ffffff;
                color: #000;
                border: 1px solid #cccccc;
                padding: 5px;
                margin: 2px;
            }

            QPushButton:hover {
                background-color: #e0e0e0;
                border: 1px solid #bcbcbc;
            }
        """
