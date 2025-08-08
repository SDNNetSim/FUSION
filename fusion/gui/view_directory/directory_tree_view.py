# pylint: disable=c-extension-no-member
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=super-with-arguments

from PyQt5 import QtWidgets, QtCore, QtGui

from fusion.gui.gui_args.config_args import SETTINGS_CONFIG_DICT

class DirectoryTreeView(QtWidgets.QTreeView):
    """
    Sets up a new directory tree view.
    """
    item_double_clicked_sig = QtCore.pyqtSignal(QtCore.QModelIndex)

    def __init__(
        self,
        file_model: QtWidgets.QFileSystemModel,
        parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.setSelectionBehavior(QtWidgets.QTreeView.SelectRows)
        self.setSelectionMode(QtWidgets.QTreeView.SingleSelection)

        self.model = file_model
        self.setModel(self.model)

        self.copied_path = None
        self.is_directory = False
        self.is_cut_operation = False

    def copy_item(
        self,
        source_index: QtCore.QModelIndex,
        is_cut_operation: bool = False
    ):
        """
        Implements the copy/cut operations in the directory tree view.
        :param source_index:        the QModelIndex of the item to be copied
        or cut.
        :param is_cut_operation:    boolean to choose between copying
        and cutting.
        """
        # Store the source path and determine if it's a directory
        self.copied_path = self.model.filePath(source_index)
        self.is_directory = QtCore.QFileInfo(self.copied_path).isDir()
        self.is_cut_operation = is_cut_operation

    def _copy_directory(
        self,
        source_dir: str,
        destination_dir: str
    ):
        """
        Copies a directory recursively

        :param source_dir:          Path to old directory location
        :param destination_dir:     Path to new directory location
        """
        source_obj = QtCore.QDir(source_dir)
        destination_obj = QtCore.QDir(destination_dir)

        destination_path = destination_obj.filePath(
            QtCore.QFileInfo(source_dir).fileName())
        if not destination_obj.exists(destination_path):
            destination_obj.mkpath(destination_path)

        for file_name in source_obj.entryList(QtCore.QDir.Files):
            QtCore.QFile.copy(source_obj.absoluteFilePath(file_name),
                              QtCore.QDir(destination_path).filePath(file_name))

        for subdir in source_obj.entryList(
                QtCore.QDir.Dirs | QtCore.QDir.NoDotAndDotDot):
            self._copy_directory(source_obj.absoluteFilePath(subdir), QtCore.QDir(destination_path).filePath(subdir))

        if self.is_cut_operation:
            self.delete_directory(source_dir)

    def paste_item(
        self,
        destination_index: QtCore.QModelIndex
    ):
        """
        Pastes a file/folder at destination_index.

        :param destination_index:    index of location where item is to be
        pasted.
        """
        if self.copied_path:
            destination_dir = self.model.filePath(destination_index)
            if not QtCore.QFileInfo(destination_dir).isDir():
                destination_dir = QtCore.QFileInfo(
                    destination_dir).absolutePath()

            if self.is_directory:
                self._copy_directory(self.copied_path, destination_dir)
            else:
                file_name = QtCore.QFileInfo(self.copied_path).fileName()
                if QtCore.QFile.copy(self.copied_path,
                                     QtCore.QDir(destination_dir).filePath(
                                         file_name)):
                    pass  # basically do nothing
                else:
                    QtWidgets.QMessageBox.critical(self, "Error",
                                                   "Failed to paste the file")

            if self.is_cut_operation:
                self._delete()
            self.refresh_view()

    def delete_item(
        self,
        target_index: QtCore.QModelIndex
    ):
        """
        Delete an item from the tree.

        :param target_index:    Index of item to be deleted.
        """
        path = self.model.filePath(target_index)
        is_directory = QtCore.QFileInfo(path).isDir()

        if is_directory:
            reply = QtWidgets.QMessageBox.question(self, "Delete Directory",
                                                   f"Are you sure you want to delete the directory '{path}' and all its contents?",
                                                   QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                   QtWidgets.QMessageBox.No)
        else:
            reply = QtWidgets.QMessageBox.question(self, "Delete File",
                                                   f"Are you sure you want to delete the file '{path}'?",
                                                   QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                   QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            if is_directory:
                QtCore.QDir(path).removeRecursively()
            else:
                QtCore.QFile.remove(path)
            self.refresh_view()  # Refresh the model to reflect the deletion

    def _delete(self):
        """
        Deletes an item after a paste operation
        """
        if self.is_directory:
            QtCore.QDir(self.copied_path).removeRecursively()
        else:
            QtCore.QFile.remove(self.copied_path)
        self.refresh_view()  # Refresh the model to reflect changes

    def handle_context_menu(
        self,
        position: QtCore.QModelIndex
    ):
        """
        Callback function to handle contextEvent signal. The context menu
        created by this function is rooted at position.

        :param position:    index of item that generated the signal.
        """
        index = self.indexAt(position)
        if index.isValid():
            menu_obj = QtWidgets.QMenu(self)

            menu_obj.addSeparator()
            copy_action = menu_obj.addAction("Copy")
            cut_action = menu_obj.addAction("Cut")
            paste_action = menu_obj.addAction("Paste")
            menu_obj.addSeparator()
            delete_action = menu_obj.addAction("Delete")
            menu_obj.addSeparator()

            action = menu_obj.exec_(self.viewport().mapToGlobal(position))

            if action == copy_action:
                self.copy_item(index)
            elif action == cut_action:
                self.copy_item(index, is_cut_operation=True)
            elif action == paste_action:
                self.paste_item(index)
            elif action == delete_action:
                self.delete_item(index)

    def refresh_view(self):
        """
        Refreshes the view of the tree model. Necessary when certain copy or cut
        operations are performed
        """
        self.setRootIndex(self.model.index(self.model.rootPath()))

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """
        Overrides mousePressEvent in QTreeView for single click

        :param event:   event instance generated by a left mouse click
                        on this object.
        """
        index = self.indexAt(event.pos())
        if event.button() == QtCore.Qt.LeftButton and index.isValid():
            self.setCurrentIndex(index)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """
        Overrides mouseDoubleClickEvent in QTreeView for double-click

        :param event:   event instance generated by a right mouse click on
                        this object.
        """
        index = self.indexAt(event.pos())
        if event.button() == QtCore.Qt.LeftButton and index.isValid():
            self.item_double_clicked_sig.emit(index)
        super().mouseDoubleClickEvent(event)
