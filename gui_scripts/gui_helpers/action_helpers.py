# pylint: disable=c-extension-no-member
# pylint: disable=duplicate-code
import sys
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QTabWidget, QWidget, QVBoxLayout # pylint: disable=no-name-in-module
from gui_scripts.gui_args.config_args import GUI_DEFAULTS

from gui_scripts.gui_helpers.general_helpers import SettingsDialog
from data_scripts.structure_data import create_network


# TODO: (version 5.5-6) We don't have to have modules end with the name "scripts"
#   - Also, we don't need every file to end with "helpers"
class NodeInfoDialog(QtWidgets.QDialog):  # pylint: disable=too-few-public-methods
    """
    Displays individual node dialog.
    """

    def __init__(self, node, info, parent=None):
        super(NodeInfoDialog, self).__init__(parent)  # pylint: disable=super-with-arguments
        self.setWindowTitle(f"Node Information - {node}")
        self.setGeometry(100, 100, 300, 200)
        self.setWindowModality(QtCore.Qt.ApplicationModal)  # Make the dialog modal
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)  # Ensure the dialog stays on top

        layout = QtWidgets.QVBoxLayout()

        info_label = QtWidgets.QLabel(f"Node: {node}\nInfo: {info}")
        layout.addWidget(info_label)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)


class TopologyCanvas(FigureCanvas):
    """
    Draws the topology canvas
    """

    def __init__(self, parent=None, width=10, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(TopologyCanvas, self).__init__(fig)  # pylint: disable=super-with-arguments
        self.setParent(parent)

        self.G = None  # pylint: disable=invalid-name

    def plot(self, G, pos):  # pylint: disable=invalid-name
        """
        Plots a single node.

        :param G: The graph.
        :param pos: Position of this node.
        """
        self.axes.clear()
        nx.draw(
            G, pos, ax=self.axes,
            with_labels=True,
            node_size=400,  # Increased node size
            font_size=10,  # Increased font size for node labels
            font_color="white",  # Set font color to white for better visibility
            font_weight="bold",  # Bold the node labels
            node_color="#00008B",  # Set node color to a darker blue (hex color)
        )
        self.axes.figure.tight_layout()  # Ensure the plot fits within the canvas
        self.draw()

    def set_picker(self, scatter):
        """
        Sets up picker events.

        :param scatter: The scatter object of the topology.
        """
        scatter.set_picker(True)
        self.mpl_connect('pick_event', self.on_pick)

    def on_pick(self, event):
        """
        Handles event to display node information on a click.

        :param event: The event object.
        """
        ind = event.ind[0]
        node = list(self.G.nodes())[ind]
        info = "Additional Info: ..."  # Replace with actual node information
        dialog = NodeInfoDialog(node, info, self.parent())
        dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        dialog.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        dialog.raise_()
        dialog.activateWindow()
        dialog.exec_()

def load_license_text(file_path: str) -> str:
    """
    Reads the license text from the specified file path.

    :param file_path: Path to the LICENSE file.
    :return: Content of the LICENSE file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "LICENSE file not found."
    except PermissionError:
        return "Permission denied when trying to read the LICENSE file."
    except OSError as e:
        return f"An OS error occurred while loading the LICENSE file: {e}"

class AboutDialog(QDialog): # pylint: disable=too-few-public-methods
    """
    A dialog showing information about the application with three tabs:
    About, License, and a blank tab for future use.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("About FUSION Simulator")
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout(self)

        # Create the tab widget
        self.tab_widget = QTabWidget(self)

        # Add tabs
        self._add_about_tab()
        self._add_license_tab()
        self._add_blank_tab()

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def _add_about_tab(self):
        """
        Create and add the 'About' tab.
        """
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)

        about_text = QtWidgets.QLabel(
            "FUSION Simulator\n\n"
            "Welcome to FUSION, an open-source venture into the future of networking.\n\n"
            "For more information, visit:"
        )
        about_text.setWordWrap(True)  # Enable word wrapping
        about_layout.addWidget(about_text)

        # Hyperlink
        link_label = QtWidgets.QLabel(
            '<a href="https://github.com/SDNNetSim/FUSION/" style="color: blue;">'
            'https://github.com/SDNNetSim/FUSION/</a>'
        )
        link_label.setOpenExternalLinks(True)  # Make the link clickable
        about_layout.addWidget(link_label)

        self.tab_widget.addTab(about_tab, "About")

    def _add_license_tab(self):
        """
        Create and add the 'License' tab.
        """
        license_tab = QWidget()
        license_layout = QVBoxLayout(license_tab)

        # Read the license text from the LICENSE file
        license_text = load_license_text(GUI_DEFAULTS['license_file_path'])  # Adjust the file path as needed
        license_text_area = QtWidgets.QPlainTextEdit(license_text)
        license_text_area.setReadOnly(True)  # Make it read-only
        license_layout.addWidget(license_text_area)

        self.tab_widget.addTab(license_tab, "License")

    def _add_blank_tab(self):
        """
        Create and add a blank tab for future use.
        """
        blank_tab = QWidget()
        blank_layout = QVBoxLayout(blank_tab)

        placeholder_label = QtWidgets.QLabel("Future functionality will go here.")
        blank_layout.addWidget(placeholder_label)

        self.tab_widget.addTab(blank_tab, "Blank Tab")


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
        settings_dialog.setStyleSheet("""
            background-color: white;
        """)
        if settings_dialog.exec() == QtWidgets.QDialog.Accepted:
            print(settings_dialog.get_settings())

    def _display_topology(self, net_name: str):
        # The new create network structure returns a tuple, we just care about the dictionary
        topology_information_dict, core_nodes_list = create_network(net_name=net_name) # pylint: disable=unused-variable

        edge_list = [(src, des, {'weight': link_len}) for (src, des), link_len in
                     topology_information_dict.items()]  # pylint: disable=no-member
        network_topo = nx.Graph(edge_list)

        pos = nx.spring_layout(network_topo, seed=5, scale=2.0)  # Adjust the scale as needed

        # Create a canvas and plot the topology
        canvas = TopologyCanvas(self.mw_topology_view_area)
        canvas.plot(network_topo, pos)
        canvas.G = network_topo  # pylint: disable=invalid-name

        # Draw nodes using scatter to enable picking
        x, y = zip(*pos.values())  # pylint: disable=invalid-name
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

        dialog_pos = self.menu_bar_obj.mapToGlobal(
            self.menu_bar_obj.rect().center()) - network_selection_dialog.rect().center()  # Center window
        network_selection_dialog.move(dialog_pos)

        network_selection_input = QtWidgets.QInputDialog()
        # TODO: Hard coded, should read the raw data directory or have a constants file
        items = ['USNet', 'NSFNet', 'Pan-European']
        net_name, valid = network_selection_input.getItem(
            network_selection_dialog, "Choose a network type:",
            "Select Network Type", items, 0, False
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
        display_topology_action = QtWidgets.QAction('&Display topology', self.menu_bar_obj)
        display_topology_action.triggered.connect(self.display_topology)
        self.menu_help_obj.file_menu_obj.addAction(display_topology_action)

    def create_save_action(self):
        """
        Create a save action to save a file.
        """
        save_action = QtWidgets.QAction('&Save', self.menu_bar_obj)
        save_action.triggered.connect(self.save_file)
        self.menu_help_obj.file_menu_obj.addAction(save_action)

    def create_exit_action(self):
        """
        Create an exit action to exit a simulation run.
        """
        exit_action = QtWidgets.QAction('&Exit', self.menu_bar_obj)
        exit_action.triggered.connect(self.menu_bar_obj.close)
        self.menu_help_obj.file_menu_obj.addAction(exit_action)

    def create_settings_action(self):
        """
        Create a settings action to trigger a display of the settings panel.
        """
        settings_action = QtWidgets.QAction('&Settings', self.menu_bar_obj)
        settings_action.triggered.connect(self.open_settings)
        self.menu_help_obj.edit_menu_obj.addAction(settings_action)

    def create_about_action(self):
        """
        Create about action to display relevant about information regarding the simulator.
        """
        about_action = QtWidgets.QAction('&About', self.menu_bar_obj)
        about_action.triggered.connect(self.show_about_dialog)
        self.menu_help_obj.help_menu_obj.addAction(about_action)

    def show_about_dialog(self):
        """
        Display the About dialog.
        """
        about_dialog = AboutDialog(self.menu_bar_obj)
        about_dialog.exec_()  # Open dialog modally


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    window.setWindowTitle('FUSION Simulator')

    action_helpers = ActionHelpers()
    action_helpers.menu_bar_obj = window.menuBar()
    action_helpers.mw_topology_view_area = QtWidgets.QScrollArea(window)
    window.setCentralWidget(action_helpers.mw_topology_view_area)

    action_helpers.display_topology()

    window.show()
    sys.exit(app.exec_())
