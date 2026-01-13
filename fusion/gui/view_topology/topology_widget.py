# pylint: disable=c-extension-no-member
# pylint: disable=duplicate-code
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore as qtc
from fusion.gui.view_topology.topology_dialog import NodeInfoDialog


# TODO: Move actions involving topology setup here.

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
