# -*- coding: utf-8 -*-
"""
Dependency Manager - Handles automatic installation of required packages
"""

import os
import sys
import subprocess
import importlib
from typing import List, Tuple, Optional
from qgis.PyQt.QtCore import QObject, pyqtSignal, QThread, QTimer
from qgis.PyQt.QtWidgets import QMessageBox, QProgressDialog, QApplication
from qgis.core import QgsMessageLog, Qgis, QgsSettings
import site


class DependencyInstaller(QThread):
    """Thread for installing dependencies without blocking UI"""
    
    progress_updated = pyqtSignal(int, str)
    installation_finished = pyqtSignal(bool, str)
    
    def __init__(self, packages: List[str]):
        super().__init__()
        self.packages = packages
        
    def run(self):
        """Install packages in separate thread"""
        try:
            total_packages = len(self.packages)
            
            for i, package in enumerate(self.packages):
                self.progress_updated.emit(
                    int((i / total_packages) * 100), 
                    f"Installing {package}..."
                )
                
                # Use pip to install package
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package, '--user'
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    error_msg = f"Failed to install {package}: {result.stderr}"
                    self.installation_finished.emit(False, error_msg)
                    return
                    
            self.progress_updated.emit(100, "Installation completed successfully!")
            self.installation_finished.emit(True, "All dependencies installed successfully")
            
        except subprocess.TimeoutExpired:
            self.installation_finished.emit(False, "Installation timed out")
        except Exception as e:
            self.installation_finished.emit(False, f"Installation error: {str(e)}")


class DependencyManager(QObject):
    """Manages plugin dependencies with automatic installation"""
    
    # Required packages with version constraints
    REQUIRED_PACKAGES = {
        'opencv-python': '4.5.0',
        'requests': '2.25.0', 
        'numpy': '1.19.0'
    }
    
    def __init__(self):
        super().__init__()
        self.settings = QgsSettings()
        self.installer_thread = None
        
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check if all required dependencies are available"""
        missing_packages = []
        
        for package_name, min_version in self.REQUIRED_PACKAGES.items():
            if not self._is_package_available(package_name, min_version):
                missing_packages.append(package_name)
                
        return len(missing_packages) == 0, missing_packages
    
    def _is_package_available(self, package_name: str, min_version: str) -> bool:
        """Check if a specific package is available with minimum version"""
        try:
            # Handle opencv-python special case
            import_name = 'cv2' if package_name == 'opencv-python' else package_name
            
            module = importlib.import_module(import_name)
            
            # Check version if available
            if hasattr(module, '__version__'):
                installed_version = module.__version__
                return self._compare_versions(installed_version, min_version) >= 0
            
            return True  # Package exists, assume compatible
            
        except ImportError:
            return False
        except Exception:
            return False
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns -1, 0, or 1"""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for v1, v2 in zip(v1_parts, v2_parts):
                if v1 < v2:
                    return -1
                elif v1 > v2:
                    return 1
            return 0
        except:
            return 0
    
    def should_auto_install(self) -> bool:
        """Check if auto-installation is enabled and not already attempted"""
        auto_install = self.settings.value("plugins/auto_georeferencer/auto_install_deps", True, type=bool)
        install_attempted = self.settings.value("plugins/auto_georeferencer/install_attempted", False, type=bool)
        
        return auto_install and not install_attempted
    
    def install_dependencies(self, missing_packages: List[str], parent_widget=None) -> bool:
        """Install missing dependencies with user consent"""
        if not missing_packages:
            return True
            
        # Ask user for permission
        msg = QMessageBox(parent_widget)
        msg.setWindowTitle("Install Required Dependencies")
        msg.setText(
            "The Auto Georeferencer plugin requires additional Python packages:\n\n" +
            "\n".join(f"• {pkg}" for pkg in missing_packages) +
            "\n\nWould you like to install them automatically?"
        )
        msg.setDetailedText(
            "These packages are required for:\n"
            "• opencv-python: Computer vision and image processing\n"
            "• requests: Downloading WMTS tiles from web services\n"
            "• numpy: Numerical array operations\n\n"
            "Installation will use 'pip install --user' and may take a few minutes."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        
        if msg.exec_() != QMessageBox.Yes:
            # Mark as attempted to avoid repeated prompts
            self.settings.setValue("plugins/auto_georeferencer/install_attempted", True)
            return False
        
        # Show progress dialog
        progress = QProgressDialog("Installing dependencies...", "Cancel", 0, 100, parent_widget)
        progress.setWindowTitle("Auto Georeferencer Setup")
        progress.setWindowModality(2)  # Application modal
        progress.show()
        
        # Start installation in separate thread
        self.installer_thread = DependencyInstaller(missing_packages)
        self.installer_thread.progress_updated.connect(
            lambda value, text: self._update_progress(progress, value, text)
        )
        self.installer_thread.installation_finished.connect(
            lambda success, message: self._installation_finished(progress, success, message, parent_widget)
        )
        
        self.installer_thread.start()
        
        # Process events while installation runs
        while self.installer_thread.isRunning():
            QApplication.processEvents()
            if progress.wasCanceled():
                self.installer_thread.terminate()
                self.installer_thread.wait()
                return False
        
        return getattr(self, '_install_success', False)
    
    def _update_progress(self, progress_dialog, value: int, text: str):
        """Update progress dialog"""
        progress_dialog.setValue(value)
        progress_dialog.setLabelText(text)
        QApplication.processEvents()
    
    def _installation_finished(self, progress_dialog, success: bool, message: str, parent_widget):
        """Handle installation completion"""
        progress_dialog.close()
        self._install_success = success
        
        # Mark installation as attempted
        self.settings.setValue("plugins/auto_georeferencer/install_attempted", True)
        
        if success:
            # Refresh site packages to make new modules available
            importlib.invalidate_caches()
            site.main()
            
            QMessageBox.information(
                parent_widget,
                "Installation Successful",
                "Dependencies installed successfully! The plugin is now ready to use."
            )
            QgsMessageLog.logMessage(
                "Auto Georeferencer dependencies installed successfully",
                "AutoGeoreferencer", Qgis.Info
            )
        else:
            QMessageBox.critical(
                parent_widget,
                "Installation Failed", 
                f"Failed to install dependencies:\n\n{message}\n\n"
                "You can try installing manually using:\n"
                "pip install opencv-python requests numpy"
            )
            QgsMessageLog.logMessage(
                f"Dependency installation failed: {message}",
                "AutoGeoreferencer", Qgis.Critical
            )
    
    def reset_installation_flag(self):
        """Reset the installation attempted flag (for testing/debugging)"""
        self.settings.setValue("plugins/auto_georeferencer/install_attempted", False)


# Global dependency manager instance
dependency_manager = DependencyManager()
