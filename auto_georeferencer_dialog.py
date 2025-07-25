# -*- coding: utf-8 -*-
"""
Auto Georeferencer Dialog - Enhanced with dependency status
"""

import os
from qgis.PyQt import uic
from qgis.PyQt.QtCore import pyqtSignal, QThread, QTimer, Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QDialogButtonBox,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QAbstractItemView,
)
from qgis.PyQt.QtGui import QPixmap, QPalette
from qgis.core import QgsProject, QgsRasterLayer, QgsCoordinateReferenceSystem
from qgis.gui import QgsProjectionSelectionWidget

from .dependency_manager import dependency_manager

# Load UI file
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'auto_georeferencer_dialog_base.ui'))


class AutoGeoreferencerDialog(QDialog, FORM_CLASS):

    def __init__(self, parent=None):
        """Constructor."""
        super(AutoGeoreferencerDialog, self).__init__(parent)
        self.setupUi(self)

        # Store multiple selected rasters
        self.selected_rasters = []
        
        # Initialize UI components
        self.setup_ui_components()
        
        # Connect signals
        self.connect_signals()
        
        # Populate raster layers
        self.populate_raster_layers()
        
        # Check and display dependency status
        self.update_dependency_status()

    def setup_ui_components(self):
        """Setup UI components with default values"""
        # Set default values
        self.spinBox_latitude.setValue(40.0)
        self.spinBox_longitude.setValue(-105.0)
        self.doubleSpinBox_boxSize.setValue(0.5)
        
        # Set output directory to user's home
        self.lineEdit_outputDir.setText(os.path.expanduser("~"))
        
        # Setup coordinate reference system selector
        self.crs_selector.setCrs(QgsCoordinateReferenceSystem("EPSG:4269"))
        
        # Set advanced options defaults
        self.spinBox_zoomLevel.setValue(14)
        self.spinBox_maxFeatures.setValue(5000)
        self.spinBox_cacheSize.setValue(50)
        self.checkBox_parallelDownload.setChecked(True)

    def connect_signals(self):
        """Connect UI signals to slots"""
        self.pushButton_browseOutput.clicked.connect(self.browse_output_directory)
        self.pushButton_refreshLayers.clicked.connect(self.populate_raster_layers)
        self.pushButton_selectRasters.clicked.connect(self.select_multiple_rasters)
        self.pushButton_checkDeps.clicked.connect(self.check_dependencies)
        
        # Advanced options toggle
        self.checkBox_showAdvanced.toggled.connect(self.toggle_advanced_options)
        
        # Auto-refresh dependency status periodically
        self.dep_timer = QTimer()
        self.dep_timer.timeout.connect(self.update_dependency_status)
        self.dep_timer.start(30000)  # Check every 30 seconds

    def toggle_advanced_options(self, show: bool):
        """Show/hide advanced options"""
        self.groupBox_advanced.setVisible(show)
        
        # Adjust dialog size
        if show:
            self.resize(self.width(), self.height() + 150)
        else:
            self.resize(self.width(), self.height() - 150)

    def populate_raster_layers(self):
        """Populate the raster layer combo box with loaded raster layers"""
        self.comboBox_rasterLayer.clear()
        
        # Get all loaded raster layers
        layers = QgsProject.instance().mapLayers().values()
        raster_layers = [layer for layer in layers if isinstance(layer, QgsRasterLayer)]
        
        if not raster_layers:
            self.comboBox_rasterLayer.addItem("No raster layers loaded")
            return
            
        for layer in raster_layers:
            self.comboBox_rasterLayer.addItem(layer.name(), layer.id())

        # Preserve selections in the multiple raster list
        if self.selected_rasters:
            current_ids = [lyr.id() for lyr in self.selected_rasters]
            for idx in range(self.comboBox_rasterLayer.count()):
                if self.comboBox_rasterLayer.itemData(idx) in current_ids:
                    self.comboBox_rasterLayer.setCurrentIndex(idx)

    def browse_output_directory(self):
        """Open file dialog to select output directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.lineEdit_outputDir.text()
        )
        
        if directory:
            self.lineEdit_outputDir.setText(directory)

    def check_dependencies(self):
        """Manually check and install dependencies"""
        deps_available, missing_packages = dependency_manager.check_dependencies()
        
        if deps_available:
            QMessageBox.information(
                self,
                "Dependencies Check",
                "All required dependencies are available!"
            )
        else:
            # Attempt installation
            if dependency_manager.install_dependencies(missing_packages, self):
                self.update_dependency_status()
            else:
                QMessageBox.warning(
                    self,
                    "Installation Failed",
                    f"Failed to install: {', '.join(missing_packages)}\n\n"
                    "Please install manually using:\n"
                    f"pip install {' '.join(missing_packages)}"
                )

    def update_dependency_status(self):
        """Update the dependency status indicator"""
        deps_available, missing_packages = dependency_manager.check_dependencies()
        
        if deps_available:
            self.label_depStatus.setText("✓ Dependencies: Available")
            self.label_depStatus.setStyleSheet("color: green; font-weight: bold;")
            self.pushButton_checkDeps.setText("Check Dependencies")
            self.pushButton_checkDeps.setEnabled(True)
        else:
            self.label_depStatus.setText(f"✗ Missing: {', '.join(missing_packages)}")
            self.label_depStatus.setStyleSheet("color: red; font-weight: bold;")
            self.pushButton_checkDeps.setText("Install Dependencies")
            self.pushButton_checkDeps.setEnabled(True)

    def get_selected_raster_layer(self):
        """Get the currently selected raster layer"""
        layer_id = self.comboBox_rasterLayer.currentData()
        if layer_id:
            return QgsProject.instance().mapLayer(layer_id)
        return None

    def select_multiple_rasters(self):
        """Open a dialog to select multiple raster layers"""
        layers = QgsProject.instance().mapLayers().values()
        raster_layers = [lyr for lyr in layers if isinstance(lyr, QgsRasterLayer)]
        if not raster_layers:
            QMessageBox.warning(self, "No Rasters", "No raster layers loaded.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Raster Layers")
        layout = QVBoxLayout(dlg)
        list_widget = QListWidget()
        list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        for lyr in raster_layers:
            item = QListWidgetItem(lyr.name())
            item.setData(Qt.UserRole, lyr.id())
            if lyr in self.selected_rasters:
                item.setSelected(True)
            list_widget.addItem(item)
        layout.addWidget(list_widget)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec_() == QDialog.Accepted:
            self.selected_rasters = [
                QgsProject.instance().mapLayer(it.data(Qt.UserRole))
                for it in list_widget.selectedItems()
            ]

    def get_selected_raster_layers(self):
        """Return selected raster layers (multi-selection)"""
        if self.selected_rasters:
            return self.selected_rasters
        layer = self.get_selected_raster_layer()
        return [layer] if layer else []

    def get_parameters(self):
        """Get all parameters from the dialog"""
        params = {
            'raster_layer': self.get_selected_raster_layer(),
            'latitude': self.spinBox_latitude.value(),
            'longitude': self.spinBox_longitude.value(),
            'box_size': self.doubleSpinBox_boxSize.value(),
            'output_directory': self.lineEdit_outputDir.text(),
            'target_crs': self.crs_selector.crs(),
            'use_manual_fallback': self.checkBox_manualFallback.isChecked(),
            'zoom_level': self.spinBox_zoomLevel.value(),
            'use_edge_detection': self.checkBox_edgeDetection.isChecked(),
            'use_clahe': self.checkBox_claheEnhancement.isChecked()
        }
        
        # Add advanced parameters if visible
        if self.groupBox_advanced.isVisible():
            params.update({
                'max_features': self.spinBox_maxFeatures.value(),
                'tile_cache_size': self.spinBox_cacheSize.value(),
                'parallel_download': self.checkBox_parallelDownload.isChecked()
            })
        
        return params

    def validate_parameters(self) -> tuple[bool, str]:
        """Validate all input parameters"""
        params = self.get_parameters()
        
        # Check raster layers
        if not self.get_selected_raster_layers():
            return False, "Please select at least one raster layer to georeference."
        
        # Check output directory
        if not os.path.exists(params['output_directory']):
            return False, "Please select a valid output directory."
        
        # Check coordinate bounds
        if not (-90 <= params['latitude'] <= 90):
            return False, "Latitude must be between -90 and 90 degrees."
        
        if not (-180 <= params['longitude'] <= 180):
            return False, "Longitude must be between -180 and 180 degrees."
        
        # Check box size
        if params['box_size'] <= 0:
            return False, "Box size must be greater than 0."
        
        # Check zoom level
        if not (1 <= params['zoom_level'] <= 18):
            return False, "Zoom level must be between 1 and 18."
        
        # Check dependencies
        deps_available, missing_packages = dependency_manager.check_dependencies()
        if not deps_available:
            return False, f"Missing dependencies: {', '.join(missing_packages)}. Please install them first."
        
        return True, ""

    def accept(self):
        """Override accept to validate parameters"""
        valid, error_msg = self.validate_parameters()
        
        if not valid:
            QMessageBox.warning(self, "Invalid Parameters", error_msg)
            return
        
        super().accept()

    def closeEvent(self, event):
        """Clean up when dialog is closed"""
        if hasattr(self, 'dep_timer'):
            self.dep_timer.stop()
        event.accept()
