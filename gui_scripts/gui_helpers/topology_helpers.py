# pylint: disable=c-extension-no-member
# pylint: disable=duplicate-code
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets as qtw, QtCore as qtc


class NodeInfoDialog(qtw.QDialog):  # pylint: disable=too-few-public-methods
    """
    Displays individual node dialog.
    """

    def __init__(self, node, info, parent=None):
        super().__init__(parent)  # pylint: disable=super-with-arguments
        self.setWindowTitle(f"Node Information - {node}")
        self.setGeometry(100, 100, 300, 200)
        self.setWindowModality(qtc.Qt.ApplicationModal)  # Make the dialog modal
        self.setWindowFlag(qtc.Qt.WindowStaysOnTopHint)  # Ensure the dialog stays on top

        layout = qtw.QVBoxLayout()

        info_label = qtw.QLabel(f"Node: {node}\nInfo: {info}")
        layout.addWidget(info_label)

        close_button = qtw.QPushButton("Close")
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
        super().__init__(fig)  # pylint: disable=super-with-arguments
        self.setParent(parent)
        self.scatter = None
        self.G = None  # pylint: disable=invalid-name

    def plot(self, graph, pos):  # pylint: disable=invalid-name
        """
        Plots a single node.

        :param graph: The graph.
        :param pos: Position of this node.
        """
        self.axes.clear()
        nx.draw(
            graph, pos, ax=self.axes,
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
        self.scatter = scatter
        scatter.set_picker(True)
        self.mpl_connect('button_press_event', self.on_pick)

    def on_pick(self, event):
        """
        Handles event to display node information on a click. If user left-clicked near
        a pick-able artifact on the plot, show NodeDialog.

        :param event: The event object.
        """
        if event.button == 1:
            contains, index = self.scatter.contains(event)
            if contains:
                ind = index['ind'][0]
                node = list(self.G.nodes())[ind]
                info = "Additional Info: ..."  # Replace with actual node information
                dialog = NodeInfoDialog(node, info, self.parent())
                dialog.setWindowModality(qtc.Qt.ApplicationModal)
                dialog.setWindowFlag(qtc.Qt.WindowStaysOnTopHint)
                dialog.show()
