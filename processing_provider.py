# -*- coding: utf-8 -*-
"""
Processing Provider for Auto Georeferencer
"""

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon
import os
from .processing_algorithm import AutoGeoreferencingAlgorithm


class AutoGeoreferencerProvider(QgsProcessingProvider):
    """Processing provider for Auto Georeferencer algorithms"""

    def __init__(self):
        QgsProcessingProvider.__init__(self)

    def unload(self):
        """Called when the provider is being unloaded"""
        pass

    def loadAlgorithms(self):
        """Load all algorithms belonging to this provider"""
        self.addAlgorithm(AutoGeoreferencingAlgorithm())

    def id(self):
        """Returns the unique provider id"""
        return 'autogeoreferencer'

    def name(self):
        """Returns the provider name"""
        return 'Auto Georeferencer'

    def icon(self):
        """Returns the provider icon"""
        return QIcon(os.path.join(os.path.dirname(__file__), 'icon.png'))

    def longName(self):
        """Returns the full provider name"""
        return 'Auto Georeferencer Processing Provider'
