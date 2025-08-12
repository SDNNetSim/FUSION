# pylint: disable=c-extension-no-member

from PyQt5 import QtWidgets
from fusion.gui.gui_args.config_args import SETTINGS_CONFIG_DICT



class SettingsDialog(QtWidgets.QDialog):  # pylint: disable=too-few-public-methods
    """
    The settings window in the menu bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings Menu")
        self.resize(400, 600)
        self.layout = QtWidgets.QVBoxLayout()
        self.tabs = QtWidgets.QTabWidget()
        self.settings_widgets = {}
        self._setup_layout()

        self.setLayout(self.layout)

    def _setup_layout(self):
        for category in SETTINGS_CONFIG_DICT:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QFormLayout()
            for setting in category["settings"]:
                widget, label = self._create_widget(setting)
                if isinstance(widget, QtWidgets.QLabel): # choosing QLabel for all headers
                    tab_layout.addRow(widget)
                else:
                    tab_layout.addRow(label, widget)
                self.settings_widgets[label] = widget
            tab.setLayout(tab_layout)
            self.tabs.addTab(tab, category["category"])
        self.layout.addWidget(self.tabs)

        self._setup_buttons()

    @staticmethod
    def _create_widget(setting):
        widget_type = setting["type"]
        widget = None
        label = setting["label"]
        if widget_type == "combo":
            widget = QtWidgets.QComboBox()
            widget.addItems(setting["options"])
            widget.setCurrentText(setting["default"])
        elif widget_type == "check":
            widget = QtWidgets.QCheckBox()
            widget.setChecked(setting["default"])
        elif widget_type == "line":
            widget = QtWidgets.QLineEdit(setting["default"])
        elif widget_type == "spin":
            widget = QtWidgets.QSpinBox()
            widget.setValue(setting["default"])
            widget.setMinimum(setting.get("min", 0))
            widget.setMaximum(setting.get("max", 100))
        elif widget_type == "double_spin":
            widget = QtWidgets.QDoubleSpinBox()
            widget.setValue(setting["default"])
            widget.setMinimum(setting.get("min", 0.0))
            widget.setSingleStep(setting.get("step", 1.0))
        elif widget_type == "header":
            widget = QtWidgets.QLabel(label)
            widget.setStyleSheet("""
                font-weight: bold;
                font-size: 13pt;
            """)
        return widget, label

    def _setup_buttons(self):
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)

    def get_settings(self):
        """
        Gets and structures all configuration settings.

        :return: The simulation configuration.
        :rtype: dict
        """
        settings = {}
        for category in SETTINGS_CONFIG_DICT:
            category_name = category["category"].lower() + "_settings"
            settings[category_name] = {}
            for setting in category["settings"]:
                label = setting["label"]
                widget = self.settings_widgets[label]
                settings[category_name][
                    self._format_label(label)] = self._get_widget_value(widget)
        return {"s1": settings}

    @staticmethod
    def _format_label(label):
        return label.lower().replace(" ", "_").replace(":", "")

    @staticmethod
    def _get_widget_value(widget):
        resp = None
        if isinstance(widget, QtWidgets.QComboBox):
            resp = widget.currentText()
        elif isinstance(widget, QtWidgets.QCheckBox):
            resp = widget.isChecked()
        elif isinstance(widget, QtWidgets.QLineEdit):
            resp = widget.text()
        elif isinstance(widget, QtWidgets.QSpinBox):
            resp = widget.value()
        elif isinstance(widget, QtWidgets.QDoubleSpinBox):
            resp = widget.value()

        return resp
