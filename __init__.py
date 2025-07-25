# -*- coding: utf-8 -*-
"""
Auto Georeferencer Plugin
"""

def classFactory(iface):
    """Load AutoGeoreferencer class from file auto_georeferencer.
    
    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .auto_georeferencer import AutoGeoreferencer
    return AutoGeoreferencer(iface)
