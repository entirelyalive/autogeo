# -*- coding: utf-8 -*-
"""
Processing Algorithm for Auto Georeferencing
"""

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterCrs,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterRasterDestination,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsCoordinateReferenceSystem
)
from .georeferencing_engine import GeoreferencingEngine, ProcessingParameters


class AutoGeoreferencingAlgorithm(QgsProcessingAlgorithm):
    """Processing algorithm for automatic georeferencing"""

    # Parameter names
    INPUT_RASTER = 'INPUT_RASTER'
    SEED_LATITUDE = 'SEED_LATITUDE'
    SEED_LONGITUDE = 'SEED_LONGITUDE'
    BOX_SIZE = 'BOX_SIZE'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    TARGET_CRS = 'TARGET_CRS'
    ZOOM_LEVEL = 'ZOOM_LEVEL'
    USE_EDGE_DETECTION = 'USE_EDGE_DETECTION'
    USE_CLAHE = 'USE_CLAHE'
    MANUAL_FALLBACK = 'MANUAL_FALLBACK'
    OUTPUT = 'OUTPUT'

    def tr(self, string):
        """Returns a translatable string with the self.tr() function."""
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        """Returns a new copy of the algorithm"""
        return AutoGeoreferencingAlgorithm()

    def name(self):
        """Returns the algorithm name"""
        return 'autogeoreference'

    def displayName(self):
        """Returns the translated algorithm name"""
        return self.tr('Auto Georeference Raster')

    def group(self):
        """Returns the name of the group this algorithm belongs to"""
        return self.tr('Georeferencing')

    def groupId(self):
        """Returns the unique ID of the group this algorithm belongs to"""
        return 'georeferencing'

    def shortHelpString(self):
        """Returns a localised short helper string for the algorithm"""
        return self.tr("""
        Automatically georeferences un-referenced raster images by matching them to a USGS topographic WMTS basemap.
        
        The algorithm:
        1. Downloads WMTS tiles covering the specified seed area
        2. Detects and matches features between the input raster and reference tiles
        3. Generates Ground Control Points (GCPs) from feature matches
        4. Warps the input raster using the GCPs
        5. Saves the georeferenced result
        
        Parameters:
        - Input Raster: The raster layer to georeference
        - Seed Latitude/Longitude: Approximate center coordinates (EPSG:4269)
        - Box Size: Half-width of the reference area in degrees
        - Target CRS: Coordinate reference system for the output
        - Zoom Level: WMTS zoom level (higher = more detail)
        """)

    def initAlgorithm(self, config=None):
        """Define the inputs and outputs of the algorithm"""
        
        # Input raster layer
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr('Input raster layer'),
                optional=False
            )
        )
        
        # Seed coordinates
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SEED_LATITUDE,
                self.tr('Seed latitude (degrees)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=40.0,
                minValue=-90.0,
                maxValue=90.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SEED_LONGITUDE,
                self.tr('Seed longitude (degrees)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=-105.0,
                minValue=-180.0,
                maxValue=180.0
            )
        )
        
        # Box size
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BOX_SIZE,
                self.tr('Half-box size (degrees)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                minValue=0.001,
                maxValue=10.0
            )
        )
        
        # Output folder
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER,
                self.tr('Output folder')
            )
        )
        
        # Target CRS
        self.addParameter(
            QgsProcessingParameterCrs(
                self.TARGET_CRS,
                self.tr('Target CRS'),
                defaultValue=QgsCoordinateReferenceSystem('EPSG:4269')
            )
        )
        
        # Zoom level
        self.addParameter(
            QgsProcessingParameterNumber(
                self.ZOOM_LEVEL,
                self.tr('WMTS zoom level'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=14,
                minValue=1,
                maxValue=18
            )
        )
        
        # Processing options
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_EDGE_DETECTION,
                self.tr('Use edge detection preprocessing'),
                defaultValue=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_CLAHE,
                self.tr('Apply CLAHE contrast enhancement'),
                defaultValue=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.MANUAL_FALLBACK,
                self.tr('Enable manual tie-point fallback'),
                defaultValue=True
            )
        )
        
        # Output
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                self.tr('Georeferenced raster')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Execute the algorithm"""
        
        # Get parameters
        input_raster = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        seed_lat = self.parameterAsDouble(parameters, self.SEED_LATITUDE, context)
        seed_lon = self.parameterAsDouble(parameters, self.SEED_LONGITUDE, context)
        box_size = self.parameterAsDouble(parameters, self.BOX_SIZE, context)
        output_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        target_crs = self.parameterAsCrs(parameters, self.TARGET_CRS, context)
        zoom_level = self.parameterAsInt(parameters, self.ZOOM_LEVEL, context)
        use_edge = self.parameterAsBool(parameters, self.USE_EDGE_DETECTION, context)
        use_clahe = self.parameterAsBool(parameters, self.USE_CLAHE, context)
        manual_fallback = self.parameterAsBool(parameters, self.MANUAL_FALLBACK, context)
        
        # Create processing parameters
        proc_params = ProcessingParameters(
            raster_layer=input_raster,
            seed_lat=seed_lat,
            seed_lon=seed_lon,
            box_size=box_size,
            output_dir=output_folder,
            target_crs=target_crs,
            zoom_level=zoom_level,
            use_edge_detection=use_edge,
            use_clahe=use_clahe,
            manual_fallback=manual_fallback
        )
        
        # Create and run georeferencing engine
        engine = GeoreferencingEngine()
        
        # Connect feedback signals
        def update_progress(progress, message):
            feedback.setProgress(progress)
            feedback.pushInfo(message)
        
        def log_message(message):
            feedback.pushInfo(message)
        
        def handle_error(error):
            feedback.reportError(error)
            raise Exception(error)
        
        engine.progress_updated.connect(update_progress)
        engine.log_message.connect(log_message)
        engine.error_occurred.connect(handle_error)
        
        # Process georeferencing
        try:
            engine.process_georeferencing(proc_params)
            
            # Return output path
            base_name = input_raster.name()
            output_filename = f"{base_name}_georeferenced.tif"
            output_path = f"{output_folder}/{output_filename}"
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            feedback.reportError(f"Processing failed: {str(e)}")
            return {}

    def flags(self):
        """Return algorithm flags"""
        return super().flags() | QgsProcessingAlgorithm.FlagNoThreading
