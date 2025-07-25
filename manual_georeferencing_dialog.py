# -*- coding: utf-8 -*-
"""
Manual Georeferencing Dialog - Fallback mode for manual tie-point selection
"""

import os
from qgis.PyQt import uic
from qgis.PyQt.QtCore import Qt, pyqtSignal, QPoint
from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidget, QTableWidgetItem, QHeaderView
from qgis.PyQt.QtGui import QPixmap, QPainter, QPen, QColor
from qgis.core import QgsPointXY
from qgis.gui import QgsMapCanvas, QgsMapToolPan, QgsMapToolZoom

# Load UI file
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'manual_georeferencing_dialog_base.ui'))


class ManualGeoreferencingDialog(QDialog, FORM_CLASS):
    """Dialog for manual tie-point selection"""
    
    # Signals
    tie_points_updated = pyqtSignal(list)  # List of tie points
    
    def __init__(self, target_image, reference_image, parent=None):
        """Constructor"""
        super(ManualGeoreferencingDialog, self).__init__(parent)
        self.setupUi(self)
        
        self.target_image = target_image
        self.reference_image = reference_image
        self.tie_points = []
        self.current_point_index = 0
        
        # Setup UI
        self.setup_image_viewers()
        self.setup_tie_point_table()
        self.connect_signals()
        
        # Instructions
        self.label_instructions.setText(
            "Click corresponding points on both images to create tie points. "
            "You need at least 3 tie points for georeferencing."
        )

    def setup_image_viewers(self):
        """Setup image viewing widgets"""
        # Convert numpy arrays to QPixmap for display
        self.target_pixmap = self.numpy_to_qpixmap(self.target_image)
        self.reference_pixmap = self.numpy_to_qpixmap(self.reference_image)
        
        # Set images to labels
        self.label_targetImage.setPixmap(self.target_pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        self.label_referenceImage.setPixmap(self.reference_pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        
        # Enable mouse tracking
        self.label_targetImage.mousePressEvent = self.target_image_clicked
        self.label_referenceImage.mousePressEvent = self.reference_image_clicked

    def setup_tie_point_table(self):
        """Setup tie point table"""
        self.tableWidget_tiePoints.setColumnCount(5)
        self.tableWidget_tiePoints.setHorizontalHeaderLabels([
            "ID", "Target X", "Target Y", "Reference X", "Reference Y"
        ])
        
        # Resize columns
        header = self.tableWidget_tiePoints.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

    def connect_signals(self):
        """Connect UI signals"""
        self.pushButton_clearPoints.clicked.connect(self.clear_tie_points)
        self.pushButton_deleteSelected.clicked.connect(self.delete_selected_point)
        self.pushButton_apply.clicked.connect(self.apply_tie_points)

    def numpy_to_qpixmap(self, np_array):
        """Convert numpy array to QPixmap"""
        import cv2
        from qgis.PyQt.QtGui import QImage
        
        # Ensure array is uint8
        if np_array.dtype != 'uint8':
            np_array = ((np_array - np_array.min()) / (np_array.max() - np_array.min()) * 255).astype('uint8')
        
        # Convert to RGB if grayscale
        if len(np_array.shape) == 2:
            np_array = cv2.cvtColor(np_array, cv2.COLOR_GRAY2RGB)
        elif len(np_array.shape) == 3 and np_array.shape[2] == 3:
            np_array = cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB)
        
        height, width, channel = np_array.shape
        bytes_per_line = 3 * width
        
        q_image = QImage(np_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)

    def target_image_clicked(self, event):
        """Handle click on target image"""
        if event.button() == Qt.LeftButton:
            # Get click position relative to image
            pos = event.pos()
            label_size = self.label_targetImage.size()
            pixmap_size = self.label_targetImage.pixmap().size()
            
            # Calculate scale factors
            scale_x = self.target_image.shape[1] / pixmap_size.width()
            scale_y = self.target_image.shape[0] / pixmap_size.height()
            
            # Calculate offset (image might be centered in label)
            offset_x = (label_size.width() - pixmap_size.width()) / 2
            offset_y = (label_size.height() - pixmap_size.height()) / 2
            
            # Convert to image coordinates
            img_x = (pos.x() - offset_x) * scale_x
            img_y = (pos.y() - offset_y) * scale_y
            
            # Store target point
            self.current_target_point = (img_x, img_y)
            self.label_status.setText(f"Target point selected: ({img_x:.1f}, {img_y:.1f}). Now click on reference image.")

    def reference_image_clicked(self, event):
        """Handle click on reference image"""
        if event.button() == Qt.LeftButton and hasattr(self, 'current_target_point'):
            # Get click position relative to image
            pos = event.pos()
            label_size = self.label_referenceImage.size()
            pixmap_size = self.label_referenceImage.pixmap().size()
            
            # Calculate scale factors
            scale_x = self.reference_image.shape[1] / pixmap_size.width()
            scale_y = self.reference_image.shape[0] / pixmap_size.height()
            
            # Calculate offset
            offset_x = (label_size.width() - pixmap_size.width()) / 2
            offset_y = (label_size.height() - pixmap_size.height()) / 2
            
            # Convert to image coordinates
            img_x = (pos.x() - offset_x) * scale_x
            img_y = (pos.y() - offset_y) * scale_y
            
            # Create tie point
            tie_point = {
                'id': len(self.tie_points) + 1,
                'target_x': self.current_target_point[0],
                'target_y': self.current_target_point[1],
                'reference_x': img_x,
                'reference_y': img_y
            }
            
            self.tie_points.append(tie_point)
            self.update_tie_point_table()
            
            # Clear current selection
            delattr(self, 'current_target_point')
            self.label_status.setText(f"Tie point {tie_point['id']} created. Click on target image for next point.")

    def update_tie_point_table(self):
        """Update the tie point table"""
        self.tableWidget_tiePoints.setRowCount(len(self.tie_points))
        
        for i, point in enumerate(self.tie_points):
            self.tableWidget_tiePoints.setItem(i, 0, QTableWidgetItem(str(point['id'])))
            self.tableWidget_tiePoints.setItem(i, 1, QTableWidgetItem(f"{point['target_x']:.1f}"))
            self.tableWidget_tiePoints.setItem(i, 2, QTableWidgetItem(f"{point['target_y']:.1f}"))
            self.tableWidget_tiePoints.setItem(i, 3, QTableWidgetItem(f"{point['reference_x']:.1f}"))
            self.tableWidget_tiePoints.setItem(i, 4, QTableWidgetItem(f"{point['reference_y']:.1f}"))

    def clear_tie_points(self):
        """Clear all tie points"""
        self.tie_points.clear()
        self.update_tie_point_table()
        self.label_status.setText("All tie points cleared. Click on target image to start.")

    def delete_selected_point(self):
        """Delete selected tie point"""
        current_row = self.tableWidget_tiePoints.currentRow()
        if current_row >= 0:
            del self.tie_points[current_row]
            # Renumber remaining points
            for i, point in enumerate(self.tie_points):
                point['id'] = i + 1
            self.update_tie_point_table()
            self.label_status.setText(f"Tie point deleted. {len(self.tie_points)} points remaining.")

    def apply_tie_points(self):
        """Apply tie points and close dialog"""
        if len(self.tie_points) < 3:
            self.label_status.setText("Error: At least 3 tie points are required for georeferencing.")
            return
        
        self.tie_points_updated.emit(self.tie_points)
        self.accept()

    def get_tie_points(self):
        """Get the current tie points"""
        return self.tie_points
