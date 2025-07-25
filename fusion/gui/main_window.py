# pylint: disable=c-extension-no-member
# pylint: disable=no-name-in-module

import sys
import multiprocessing
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileSystemModel, QTabWidget, QPlainTextEdit
from fusion.gui.gui_helpers.menu_helpers import MenuHelpers
from fusion.gui.gui_helpers.action_helpers import ActionHelpers
from fusion.gui.gui_helpers.button_helpers import ButtonHelpers
from fusion.gui.gui_helpers.highlight_helpers import PythonHighlighter
from fusion.gui.gui_helpers.general_helpers import DirectoryTreeView
from fusion.gui.gui_args.style_args import STYLE_SHEET


class MainWindow(QtWidgets.QMainWindow):
    """
    The main window class, central point that controls all GUI functionality and actions.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FUSION Simulator")
        self.resize(1280, 720)

        self.menu_help_obj = MenuHelpers()
        self.ac_help_obj = ActionHelpers()
        self.button_help_obj = ButtonHelpers()

        self.progress_bar = QtWidgets.QProgressBar()
        self.simulation_config = None  # Will hold simulation configuration (sims_dict)
        self.shared_progress_dict = None  # Will hold the Manager dictionary
        self.progress_values = None
        self.progress_queue = multiprocessing.Queue()
        self.log_queue = None
        self.total_simulations = None
        self.global_work_units = None

        # Other GUI variables...
        self.current_file_path = None
        self.main_widget = None
        self.main_layout = None
        self.horizontal_splitter = None
        self.first_info_layout = None
        self.directory_tree_obj = None
        self.tab_widget = None
        self.file_editor = None
        self.mw_topology_view_area = None
        self.vertical_splitter = None
        self.first_info_pane = None
        self.bottom_pane = None
        self.menu_bar = None
        self.tool_bar = None
        self.status_bar = None
        self.highlighter = None

        self.project_directory = QtCore.QDir.currentPath()
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(self.project_directory)

        self.init_ui()
        self.init_menu_bar()
        self.init_tool_bar()
        self.init_status_bar()
        self.apply_styles()

        # MULTI-PROCESS PROGRESS POLLING:
        self.progress_timer = QtCore.QTimer(self)
        self.progress_timer.timeout.connect(self.poll_progress)
        self.progress_timer.start(500)  # Poll every 500 ms

        self.log_timer = QtCore.QTimer(self)
        self.log_timer.timeout.connect(self.poll_log_queue)
        self.log_timer.start(500)  # Adjust polling interval as needed

    def poll_log_queue(self):
        """
        Checks the log queue for new messages and appends them to bottom pane.
        """
        if hasattr(self, 'log_queue'):
            while not self.log_queue.empty():
                message = self.log_queue.get()
                self.bottom_pane.appendPlainText(message)

    def init_ui(self):
        """
        Initialize the main user-interface.
        """
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.horizontal_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.first_info_pane = QtWidgets.QWidget()
        self.first_info_layout = QtWidgets.QVBoxLayout(self.first_info_pane)
        self.directory_tree_obj = DirectoryTreeView(self.file_model)
        self.directory_tree_obj.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.directory_tree_obj.customContextMenuRequested.connect(self.directory_tree_obj.handle_context_menu)
        self.directory_tree_obj.setRootIndex(self.file_model.index(self.project_directory))
        self.directory_tree_obj.item_double_clicked_sig.connect(self.on_tree_item_dclicked)
        self.directory_tree_obj.setStyleSheet("font-size: 12pt;")
        self.directory_tree_obj.setHeaderHidden(True)
        self.directory_tree_obj.setColumnHidden(1, True)
        self.directory_tree_obj.setColumnHidden(2, True)
        self.directory_tree_obj.setColumnHidden(3, True)
        self.first_info_layout.addWidget(self.directory_tree_obj)
        self.horizontal_splitter.addWidget(self.first_info_pane)

        self.tab_widget = QTabWidget()
        self.file_editor = QtWidgets.QTextEdit()
        self.file_editor.setObjectName("src_code_editor")
        self.file_editor.setStyleSheet("font-size: 12pt;")
        self.tab_widget.addTab(self.file_editor, "File Editor")
        self.highlighter = PythonHighlighter(self.file_editor.document())

        self.mw_topology_view_area = QtWidgets.QScrollArea()
        init_topology_data = QtWidgets.QLabel("Nothing to display", self.mw_topology_view_area)
        init_topology_data.setStyleSheet("font-size: 11pt")
        init_topology_data.setAlignment(QtCore.Qt.AlignCenter)
        self.mw_topology_view_area.setWidget(init_topology_data)
        self.mw_topology_view_area.setAlignment(QtCore.Qt.AlignCenter)
        self.mw_topology_view_area.setWidgetResizable(True)
        self.tab_widget.setStyleSheet("font-size: 12pt")
        self.tab_widget.addTab(self.mw_topology_view_area, "Topology View")
        self.horizontal_splitter.addWidget(self.tab_widget)

        self.vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vertical_splitter.addWidget(self.horizontal_splitter)

        self.bottom_pane = QPlainTextEdit(self)
        self.bottom_pane.setReadOnly(True)
        self.bottom_pane.setMinimumHeight(150)
        self.vertical_splitter.addWidget(self.bottom_pane)
        self.main_layout.addWidget(self.vertical_splitter, stretch=1)

    def on_tree_item_dclicked(self, index):
        """
        Performs an action when treeview is double-clicked.

        :param index: Index of file path displayed in the tree.
        """
        file_path = self.file_model.filePath(index)
        if QtCore.QFileInfo(file_path).isFile():
            file_index = self.tab_widget.indexOf(self.file_editor)
            self.tab_widget.setTabText(file_index, QtCore.QFileInfo(file_path).fileName())
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.file_editor.setPlainText(content)
            self.current_file_path = file_path
            self.tab_widget.setCurrentWidget(self.file_editor)

    def save_file(self):
        """
        Saves a file edited.
        """
        if hasattr(self, 'current_file_path') and self.current_file_path:
            with open(self.current_file_path, 'w', encoding='utf-8') as file:
                file.write(self.file_editor.toPlainText())

    def init_menu_bar(self):
        """
        Initialize the menu bar.
        """
        self.menu_bar = self.menuBar()
        self.menu_help_obj.menu_bar_obj = self.menu_bar
        self.menu_help_obj.create_file_menu()
        self.menu_help_obj.create_edit_menu()
        self.menu_help_obj.create_help_menu()
        self.ac_help_obj.mw_topology_view_area = self.mw_topology_view_area
        self.ac_help_obj.menu_help_obj = self.menu_help_obj
        self.ac_help_obj.menu_bar_obj = self.menu_bar
        self.ac_help_obj.create_topology_action()
        self.ac_help_obj.create_save_action()
        exit_action = QtWidgets.QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        self.menu_help_obj.file_menu_obj.addAction(exit_action)
        self.ac_help_obj.create_settings_action()
        self.ac_help_obj.create_about_action()

    def init_tool_bar(self):
        """
        Initialize the toolbar.
        """
        self.tool_bar = self.addToolBar('Main Toolbar')
        self.tool_bar.setMovable(False)
        self.tool_bar.setIconSize(QtCore.QSize(15, 15))
        save_action = QtWidgets.QAction('Save', self)
        save_action.triggered.connect(self.save_file)
        self.tool_bar.addAction(save_action)
        self.button_help_obj.bottom_right_pane = self.bottom_pane
        self.button_help_obj.progress_bar = self.progress_bar
        self.button_help_obj.create_settings_button()
        self.button_help_obj.create_start_button()
        self.button_help_obj.create_stop_button()
        self.button_help_obj.create_pause_button()
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.button_help_obj.start_button)
        self.tool_bar.addAction(self.button_help_obj.pause_button)
        self.tool_bar.addAction(self.button_help_obj.stop_button)
        self.tool_bar.addSeparator()
        self.tool_bar.addWidget(self.button_help_obj.settings_button)

    def init_status_bar(self):
        """
        Initialize the status bar.
        """
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('Active')
        self.status_bar.addWidget(self.progress_bar)
        self.progress_bar.setRange(0, 1000)  # Set range to match simulation values
        self.progress_bar.setVisible(False)

    def apply_styles(self):
        """
        Apply styles to the display.
        """
        self.setStyleSheet(STYLE_SHEET)

    def set_shared_progress_dict(self, progress_dict):
        """
        Sets the shared progress dictionary and records the fixed number of simulations.
        """
        self.shared_progress_dict = progress_dict
        self.total_simulations = len(progress_dict)
        print("Total simulations (from GUI):", self.total_simulations)

    def poll_progress(self):
        """
        Called by a QTimer every 500 ms. Drains the progress_queue of (thread_id, done_units) updates.
        The sum of done_units across all threads is divided by self.global_work_units to get a fraction.
        That fraction is turned into a 0..1000 scale for the progress bar.
        """
        # If we haven't set up the queue yet, nothing to do.
        if not hasattr(self, 'progress_queue'):
            return

        # Drain the queue fully
        while not self.progress_queue.empty():
            thread_id, done_units = self.progress_queue.get()
            # Overwrite the completed units for that child
            self.progress_values[thread_id] = done_units

        # Sum across all children
        total_done = sum(self.progress_values.values())

        # If for some reason global_work_units is zero, guard to avoid division by zero
        if getattr(self, 'global_work_units', 0) == 0:
            fraction = 0.0
        else:
            fraction = total_done / self.global_work_units

        global_progress = int(fraction * 1000)

        #print(f"poll_progress => progress_values: {dict(self.progress_values)}")
        #print(f"poll_progress => total_done={total_done}, fraction={fraction:.3f}, global_progress={global_progress}")

        # Animate the progress bar to the new global_progress
        self.button_help_obj.update_progress(global_progress)

        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)

    def set_simulation_config(self, config):
        """
        Stores the simulation configuration and sets up two queues for progress and logs.
        Also calculates the global total of iteration units across all processes
        so we know how to map partial completions to a 0..1000 scale.
        """

        self.simulation_config = config

        # Queues for child processes to send backlogs or progress
        self.log_queue = multiprocessing.Queue()

        # This dict tracks how many iteration units each process has completed so far.
        self.progress_values = {}

        # Compute how many total iteration units exist across ALL processes.
        # Each process might run multiple Erlang volumes, each with a certain number of iterations.
        total_work_units = 0
        for key, conf in self.simulation_config.items(): # pylint: disable=unused-variable
            # Unified access to Erlang start/stop/step
            if 'erlangs' in conf:
                erlangs = conf['erlangs']
            else:
                erlangs = {
                    'start': conf['erlang_start'],
                    'stop': conf['erlang_stop'],
                    'step': conf['erlang_step']
                }

            start, stop, step = erlangs['start'], erlangs['stop'], erlangs['step']
            count_erlangs = len(range(start, stop, step))
            max_iters = conf['max_iters']
            total_work_units += (count_erlangs * max_iters)

        # Store that globally in the GUI, used by poll_progress()
        self.global_work_units = total_work_units

        # Insert the queues into each process config.
        # They will push logs & partial iteration counts to these queues.
        for key, conf in self.simulation_config.items():
            conf['progress_queue'] = self.progress_queue
            conf['log_queue'] = self.log_queue

        # Also store them on the button helpers if needed.
        if self.button_help_obj is not None:
            self.button_help_obj.simulation_config = self.simulation_config
            self.button_help_obj.log_queue = self.log_queue
            self.button_help_obj.progress_queue = self.progress_queue


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
