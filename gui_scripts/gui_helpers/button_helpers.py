# button_helpers.py
import os
import multiprocessing
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve
from PyQt5 import QtWidgets, QtGui, QtCore
from gui_scripts.gui_helpers.general_helpers import SettingsDialog
from gui_scripts.gui_helpers.general_helpers import SimulationThread

# We import the simulation runner's run() function.
from run_sim import run

class ButtonHelpers:
    def __init__(self):
        self.simulation_thread = None
        self.bottom_right_pane = None
        self.progress_bar = None
        self.start_button = None
        self.pause_button = None
        self.stop_button = None
        self.settings_button = None
        self.simulation_process = None
        self.media_dir = 'media'
        self.simulation_config = None  # To be set by MainWindow
        self.shared_progress_dict = None  # To be set by MainWindow
        self.stop_flag = multiprocessing.Event()  # Shared flag for stopping the simulation

    def output_hints(self, message: str):
        self.bottom_right_pane.appendPlainText(message)

    def update_progress(self, new_value: int):
        """
        Animates the progress bar smoothly from its current value to new_value.
        """
        # Stop any currently running animation.
        if hasattr(self, 'progress_anim') and self.progress_anim is not None:
            self.progress_anim.stop()
        # Create a new animation for the progress bar's "value" property.
        self.progress_anim = QPropertyAnimation(self.progress_bar, b"value")
        self.progress_anim.setDuration(500)  # Adjust duration (in ms) as needed.
        self.progress_anim.setStartValue(self.progress_bar.value())
        self.progress_anim.setEndValue(new_value)
        # Use an easing curve for a more natural, smooth transition.
        self.progress_anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.progress_anim.start()

    def simulation_finished(self):
        # Do not hide or reset the progress bar immediately,
        # so the final progress (1000) remains visible.
        # self.progress_bar.setVisible(False)
        # self.progress_bar.setValue(0)
        self.simulation_thread = None

    def setup_simulation_thread(self):
        # In this multi-process mode, we will not use SimulationThread,
        # so we do not set it up here.
        pass

    def start_simulation(self):
        """
        Starts the simulation in a separate process (rather than a separate thread),
        ensuring that the multiprocessing.Manager dictionary is shared properly.
        """
        import multiprocessing
        from run_sim import run

        self.bottom_right_pane.clear()

        if self.simulation_config is None:
            print("Error: simulation configuration is not set!")
            return

        self.stop_flag.clear()  # Clear the stop flag before starting the simulation

        # Create and start a separate process that runs the simulations
        sim_process = multiprocessing.Process(
            target=run,
            kwargs={'sims_dict': self.simulation_config, 'stop_flag': self.stop_flag}
        )
        sim_process.start()

        self.simulation_thread = SimulationThread()
        self.simulation_thread.output_hints_signal.connect(self.output_hints)
        self.simulation_thread.progress_changed.connect(self.update_progress)
        self.simulation_thread.finished_signal.connect(self.simulation_finished)
        self.simulation_thread.finished.connect(self.simulation_thread.deleteLater)
        self.simulation_thread.start()

        self.simulation_process = sim_process
        print("Multiprocess simulation launched directly.")
        self.start_button.setText("Start")

    def pause_simulation(self):
        if self.simulation_process and self.simulation_process.is_alive():
            # Pause/resume logic would need to be implemented;
            # in a multiprocessing.Process, pausing isn't trivial.
            self.start_button.setText("Resume")

    def stop_simulation(self):
        if self.simulation_process and self.simulation_process.is_alive():
            self.stop_flag.set()  # Set the stop flag to signal the simulation process to stop
            self.simulation_process.join()  # Wait for the simulation process to finish
            self.simulation_process = None

        # Reset the progress bar and other relevant state variables
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.bottom_right_pane.clear()
        self.simulation_config = None  # Clear the simulation configuration
        self.shared_progress_dict = None  # Clear the shared progress dictionary

        self.start_button.setText("Start")

    def create_start_button(self):
        self.start_button = QtWidgets.QAction()
        resource_name = "light-green-play-button.png"
        self.media_dir = os.path.join('gui_scripts', 'media')
        self.start_button.setIcon(QtGui.QIcon(os.path.join(os.getcwd(), self.media_dir, resource_name)))
        self.start_button.setText("Start")
        self.start_button.triggered.connect(self.start_simulation)

    def create_pause_button(self):
        self.pause_button = QtWidgets.QAction()
        resource_name = "pause.png"
        self.pause_button.setIcon(QtGui.QIcon(os.path.join(os.getcwd(), self.media_dir, resource_name)))
        self.pause_button.setText("Pause")
        self.pause_button.triggered.connect(self.pause_simulation)

    def create_stop_button(self):
        self.stop_button = QtWidgets.QAction()
        resource_name = "light-red-stop-button.png"
        self.stop_button.setIcon(QtGui.QIcon(os.path.join(os.getcwd(), self.media_dir, resource_name)))
        self.stop_button.setText("Stop")
        self.stop_button.triggered.connect(self.stop_simulation)

    @staticmethod
    def open_settings():
        settings_dialog = SettingsDialog()
        settings_dialog.setModal(True)
        settings_dialog.setStyleSheet("background-color: white;")
        if settings_dialog.exec() == QtWidgets.QDialog.Accepted:
            print(settings_dialog.get_settings())

    def create_settings_button(self):
        self.settings_button = QtWidgets.QToolButton()
        resource_name = "gear.png"
        self.settings_button.setIcon(QtGui.QIcon(os.path.join(os.getcwd(), self.media_dir, resource_name)))
        self.settings_button.setText("Settings")
        self.settings_button.setStyleSheet("background-color: transparent;")
        self.settings_button.clicked.connect(self.open_settings)
