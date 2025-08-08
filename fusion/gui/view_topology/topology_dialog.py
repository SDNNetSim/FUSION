# pylint: disable=c-extension-no-member
# pylint: disable=duplicate-code
from PyQt5 import QtWidgets, QtCore


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
