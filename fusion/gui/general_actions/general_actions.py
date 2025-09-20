# pylint: disable=c-extension-no-member
# pylint: disable=duplicate-code
import sys
from idlelib.help_about import AboutDialog

import networkx as nx
from PyQt5 import QtWidgets

from fusion.gui.general_dialogs.settings import SettingsDialog
from fusion.gui.view_topology.topology_widget import TopologyCanvas
from fusion.io.structure import create_network

# TODO: It looks like actions within this class should be split between their appropriate sections
# TODO: Remove show_about_dialog if unused elsewhere


class ActionHelpers:
    """
    Contains methods related to performing actions.
    """

    def __init__(self):
        self.menu_bar_obj = None  # Updated from run_gui.py script
        self.menu_help_obj = None  # Created in menu_helpers.py
        self.mw_topology_view_area = None  # Updated from the run_gui.py script

    @staticmethod
    def save_file():
        """
        Saves a file.
        """
        print("Save file action triggered.")

    @staticmethod
    def about():
        """
        Shows about dialog.
        """
        print("Show about dialog.")

    @staticmethod
    def open_settings():
        """
        Opens the settings panel.
        """
        settings_dialog = SettingsDialog()
        settings_dialog.setModal(True)
        settings_dialog.setStyleSheet(
            """
            background-color: white;
        """
        )
        if settings_dialog.exec() == QtWidgets.QDialog.Accepted:
            print(settings_dialog.get_settings())

    def _display_topology(self, net_name: str):
        # The new create network structure returns a tuple, we just care about the dictionary
        topology_information_dict, core_nodes_list = create_network(
            net_name=net_name
        )  # pylint: disable=unused-variable

        edge_list = [
            (src, des, {"weight": link_len})
            for (src, des), link_len in topology_information_dict.items()
        ]  # pylint: disable=no-member
        network_topo = nx.Graph(edge_list)

        pos = nx.spring_layout(
            network_topo, seed=5, scale=2.0
        )  # Adjust the scale as needed

        # Create a canvas and plot the topology
        canvas = TopologyCanvas(self.mw_topology_view_area)
        canvas.plot(network_topo, pos)
        canvas.G = network_topo  # pylint: disable=invalid-name

        # Draw nodes using scatter to enable picking
        x, y = zip(*pos.values(), strict=False)  # pylint: disable=invalid-name
        scatter = canvas.axes.scatter(x, y, s=200)
        canvas.set_picker(scatter)

        self.mw_topology_view_area.setWidget(canvas)
        print("Topology displayed")  # Debugging line

    def display_topology(self):
        """
        Displays a network topology.
        """
        network_selection_dialog = QtWidgets.QDialog()
        network_selection_dialog.setSizeGripEnabled(True)

        dialog_pos = (
            self.menu_bar_obj.mapToGlobal(self.menu_bar_obj.rect().center())
            - network_selection_dialog.rect().center()
        )  # Center window
        network_selection_dialog.move(dialog_pos)

        network_selection_input = QtWidgets.QInputDialog()
        # TODO: Hard coded, should read the raw data directory or have a constants file
        items = ["USNet", "NSFNet", "Pan-European"]
        net_name, valid = network_selection_input.getItem(
            network_selection_dialog,
            "Choose a network type:",
            "Select Network Type",
            items,
            0,
            False,
        )

        # we should really only be checking if valid is true
        # if true then user must have provided a valid name since
        # we give users only three choices anyway. Otherwise, do nothing
        if valid:
            self._display_topology(net_name=net_name)

    def create_topology_action(self):
        """
        Creates the action to display a topology properly.
        """
        display_topology_action = QtWidgets.QAction(
            "&Display topology", self.menu_bar_obj
        )
        display_topology_action.triggered.connect(self.display_topology)
        self.menu_help_obj.file_menu_obj.addAction(display_topology_action)

    def create_save_action(self):
        """
        Create a save action to save a file.
        """
        save_action = QtWidgets.QAction("&Save", self.menu_bar_obj)
        save_action.triggered.connect(self.save_file)
        self.menu_help_obj.file_menu_obj.addAction(save_action)

    def create_exit_action(self):
        """
        Create an exit action to exit a simulation run.
        """
        exit_action = QtWidgets.QAction("&Exit", self.menu_bar_obj)
        exit_action.triggered.connect(self.menu_bar_obj.close)
        self.menu_help_obj.file_menu_obj.addAction(exit_action)

    def create_settings_action(self):
        """
        Create a settings action to trigger a display of the settings panel.
        """
        settings_action = QtWidgets.QAction("&Settings", self.menu_bar_obj)
        settings_action.triggered.connect(self.open_settings)
        self.menu_help_obj.edit_menu_obj.addAction(settings_action)

    def create_about_action(self):
        """
        Create about action to display relevant about information regarding the simulator.
        """
        about_action = QtWidgets.QAction("&About", self.menu_bar_obj)
        about_action.triggered.connect(self.show_about_dialog)
        self.menu_help_obj.help_menu_obj.addAction(about_action)

    def show_about_dialog(self):
        """
        Display the About dialog.
        """
        about_dialog = AboutDialog(self.menu_bar_obj)
        about_dialog.exec_()  # pylint: disable=no-member


def load_license_text(file_path: str) -> str:
    """
    Reads the license text from the specified file path.

    :param file_path: Path to the LICENSE file.
    :return: Content of the LICENSE file as a string.
    """
    try:
        with open(file_path, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "LICENSE file not found."
    except PermissionError:
        return "Permission denied when trying to read the LICENSE file."
    except OSError as e:
        return f"An OS error occurred while loading the LICENSE file: {e}"


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("FUSION Simulator")

    action_helpers = ActionHelpers()
    action_helpers.menu_bar_obj = window.menuBar()
    action_helpers.mw_topology_view_area = QtWidgets.QScrollArea(window)
    window.setCentralWidget(action_helpers.mw_topology_view_area)

    action_helpers.display_topology()

    window.show()
    sys.exit(app.exec_())
