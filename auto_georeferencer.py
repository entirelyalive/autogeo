# -*- coding: utf-8 -*-
"""
Auto Georeferencer Plugin Main Class - Enhanced with dependency management
"""

import os
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt, QThread, pyqtSignal, QTimer
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox, QProgressDialog
from qgis.core import QgsProject, QgsApplication, QgsProcessingAlgRunnerTask, Qgis
from qgis.gui import QgsMessageBar

from .resources import *
from .dependency_manager import dependency_manager
from .auto_georeferencer_dialog import AutoGeoreferencerDialog
from .processing_provider import AutoGeoreferencerProvider


class AutoGeoreferencer:
    """QGIS Plugin Implementation with automatic dependency management."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        
        # Initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'AutoGeoreferencer_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Auto Georeferencer')
        self.toolbar = self.iface.addToolBar(u'AutoGeoreferencer')
        self.toolbar.setObjectName(u'AutoGeoreferencer')
        
        # Check if plugin was started the first time in current QGIS session
        self.first_start = None
        
        # Initialize processing provider
        self.provider = None
        
        # Dependencies check flag
        self.dependencies_checked = False

    def tr(self, message):
        """Get the translation for a string using Qt translation API."""
        return QCoreApplication.translate('AutoGeoreferencer', message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar."""

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToRasterMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/auto_georeferencer/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Auto Georeferencer'),
            callback=self.run,
            parent=self.iface.mainWindow())

        # Will be set False in run()
        self.first_start = True
        
        # Initialize processing provider
        self.initProcessing()
        
        # Schedule dependency check for after QGIS fully loads
        QTimer.singleShot(2000, self.check_dependencies_delayed)

    def check_dependencies_delayed(self):
        """Check dependencies after QGIS has fully loaded"""
        if not self.dependencies_checked:
            self.dependencies_checked = True
            
            # Check if dependencies are available
            deps_available, missing_packages = dependency_manager.check_dependencies()
            
            if not deps_available and dependency_manager.should_auto_install():
                # Show info message about dependency installation
                self.iface.messageBar().pushMessage(
                    "Auto Georeferencer",
                    f"Missing dependencies: {', '.join(missing_packages)}. Click the plugin to install them.",
                    level=Qgis.Warning,
                    duration=10
                )

    def initProcessing(self):
        """Create the Processing provider"""
        self.provider = AutoGeoreferencerProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginRasterMenu(
                self.tr(u'&Auto Georeferencer'),
                action)
            self.iface.removeToolBarIcon(action)
        
        # Remove toolbar
        del self.toolbar
        
        # Remove processing provider
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)

    def run(self):
        """Run method that performs all the real work"""
        
        # Check dependencies first
        deps_available, missing_packages = dependency_manager.check_dependencies()
        
        if not deps_available:
            # Attempt to install missing dependencies
            if not dependency_manager.install_dependencies(missing_packages, self.iface.mainWindow()):
                # Installation failed or cancelled
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "Dependencies Required",
                    "The Auto Georeferencer plugin requires additional Python packages to function.\n\n"
                    "Please install them manually using:\n"
                    f"pip install {' '.join(missing_packages)}\n\n"
                    "Then restart QGIS."
                )
                return
            
            # Re-check dependencies after installation
            deps_available, missing_packages = dependency_manager.check_dependencies()
            if not deps_available:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Installation Verification Failed",
                    "Dependencies were installed but are still not available. "
                    "Please restart QGIS and try again."
                )
                return

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = AutoGeoreferencerDialog()

        # Show the dialog
        self.dlg.show()
        result = self.dlg.exec_()
        
        if result:
            # User clicked OK, process the georeferencing
            self.process_georeferencing()
    
    def process_georeferencing(self):
        """Process georeferencing with the selected parameters"""
        try:
            from .georeferencing_engine import GeoreferencingEngine, ProcessingParameters
            
            # Get parameters from dialog
            params_dict = self.dlg.get_parameters()
            raster_layers = self.dlg.get_selected_raster_layers()

            # Validate parameters
            if not raster_layers:
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "Invalid Parameters",
                    "Please select at least one raster layer to georeference."
                )
                return
            
            if not os.path.exists(params_dict['output_directory']):
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "Invalid Parameters", 
                    "Please select a valid output directory."
                )
                return
            
            # Create and configure processing engine once
            self.engine = GeoreferencingEngine()

            # Connect engine signals
            self.engine.progress_updated.connect(self.update_progress)
            self.engine.log_message.connect(self.log_message)
            self.engine.error_occurred.connect(self.handle_error)
            self.engine.processing_finished.connect(self.processing_finished)
            self.engine.manual_mode_requested.connect(self.handle_manual_mode)

            for idx, layer in enumerate(raster_layers, start=1):
                proc_params = ProcessingParameters(
                    raster_layer=layer,
                    seed_lat=params_dict['latitude'],
                    seed_lon=params_dict['longitude'],
                    box_size=params_dict['box_size'],
                    output_dir=params_dict['output_directory'],
                    target_crs=params_dict['target_crs'],
                    zoom_level=params_dict['zoom_level'],
                    use_edge_detection=params_dict.get('use_edge_detection', True),
                    use_clahe=params_dict.get('use_clahe', True),
                    manual_fallback=params_dict['use_manual_fallback']
                )

                # Create progress dialog per raster
                self.progress_dialog = QProgressDialog(
                    f"Processing {layer.name()} ({idx}/{len(raster_layers)})...",
                    "Cancel", 0, 100, self.iface.mainWindow()
                )
                self.progress_dialog.setWindowTitle("Auto Georeferencer")
                self.progress_dialog.setWindowModality(Qt.WindowModal)
                self.progress_dialog.show()

                # Start processing
                self.engine.process_georeferencing(proc_params)
            
        except Exception as e:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Processing Error",
                f"An error occurred while starting the georeferencing process:\n\n{str(e)}"
            )
    
    def update_progress(self, value: int, message: str):
        """Update progress dialog"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(message)
            
            # Update log in dialog if available
            if hasattr(self.dlg, 'textEdit_log'):
                self.dlg.textEdit_log.append(f"[{value}%] {message}")
    
    def log_message(self, message: str):
        """Log message to dialog and QGIS log"""
        if hasattr(self.dlg, 'textEdit_log'):
            self.dlg.textEdit_log.append(message)
        
        from qgis.core import QgsMessageLog, Qgis
        QgsMessageLog.logMessage(message, "AutoGeoreferencer", Qgis.Info)
    
    def handle_error(self, error_message: str):
        """Handle processing errors"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        
        QMessageBox.critical(
            self.iface.mainWindow(),
            "Georeferencing Error",
            f"An error occurred during processing:\n\n{error_message}"
        )
        
        if hasattr(self.dlg, 'textEdit_log'):
            self.dlg.textEdit_log.append(f"ERROR: {error_message}")
    
    def processing_finished(self, success: bool, output_path: str):
        """Handle processing completion"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        
        if success:
            QMessageBox.information(
                self.iface.mainWindow(),
                "Georeferencing Complete",
                f"Georeferencing completed successfully!\n\nOutput saved to:\n{output_path}"
            )
            
            # Show success message in message bar
            self.iface.messageBar().pushMessage(
                "Auto Georeferencer",
                f"Successfully georeferenced image: {os.path.basename(output_path)}",
                level=Qgis.Success,
                duration=5
            )
        else:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Georeferencing Failed",
                "Georeferencing process failed. Please check the log for details."
            )
    
    def handle_manual_mode(self):
        """Handle switch to manual tie-point mode"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        
        try:
            from .manual_georeferencing_dialog import ManualGeoreferencingDialog
            
            # Create manual georeferencing dialog
            manual_dialog = ManualGeoreferencingDialog(
                self.engine.target_image,
                self.engine.reference_image,
                self.iface.mainWindow()
            )
            
            # Connect tie points signal
            manual_dialog.tie_points_updated.connect(self.handle_manual_tie_points)
            
            # Show dialog
            result = manual_dialog.exec_()
            
            if result == manual_dialog.Accepted:
                tie_points = manual_dialog.get_tie_points()
                self.handle_manual_tie_points(tie_points)
            
        except Exception as e:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Manual Mode Error",
                f"Failed to open manual georeferencing mode:\n\n{str(e)}"
            )
    
    def handle_manual_tie_points(self, tie_points):
        """Process manual tie points"""
        if len(tie_points) < 3:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Insufficient Tie Points",
                "At least 3 tie points are required for georeferencing."
            )
            return
        
        try:
            # Convert tie points to GCPs and continue processing
            # This would integrate with the existing engine
            QMessageBox.information(
                self.iface.mainWindow(),
                "Manual Tie Points",
                f"Received {len(tie_points)} tie points. Manual processing not yet implemented."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Manual Processing Error",
                f"Failed to process manual tie points:\n\n{str(e)}"
            )
