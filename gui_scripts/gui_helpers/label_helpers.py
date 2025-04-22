# pylint: disable=c-extension-no-member
# pylint: disable=no-name-in-module
# pylint: disable=super-with-arguments

from PyQt5 import QtWidgets as qtw, QtCore as qtc, QtGui as qtg


class HoverLabel(qtw.QLabel):
    """
    Handles all labels for hover actions.
    """
    hover_changed = qtc.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setToolTip("This is the data displayed on hover.")
        self.setSizePolicy(qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Minimum)
        self.setMaximumSize(60, 60)

        self.icon = None
        self.pixmap = None

    def set_icon(self, icon_path, size=qtc.QSize(64, 64)):
        """
        Sets a specific icon.
        :param icon_path: The icon location.
        :param size: Desired size of the icon.
        """
        self.icon = qtg.QIcon(icon_path)
        self.pixmap = self.icon.pixmap(size)
        self.setPixmap(self.pixmap)
        self.resize(self.pixmap.size())

    def enter_event(self, event):
        """
        Enters an event via text.

        :param event: The event to be emitted.
        """
        self.setText(self.hoverText)
        self.hover_changed.emit(True)
        super().enterEvent(event)

    def leave_event(self, event):
        """
        Leaves an event.

        :param event: The event to remove.
        """
        self.setText(self.normalText)
        self.hover_changed.emit(False)
        super().leaveEvent(event)

    def update_tool_tip(self, new_data):
        """
        Updates to a tool tip text.

        :param new_data: Text to update with.
        """
        tool_tip_text = f"Dynamic Data: {new_data}"
        self.setToolTip(tool_tip_text)
