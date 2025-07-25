# -*- coding: utf-8 -*-
"""
Georeferencing Engine - Enhanced with performance optimizations
"""

import os
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import threading
import concurrent.futures
from functools import lru_cache

from qgis.PyQt.QtCore import QObject, pyqtSignal, QThread
from qgis.core import (
    QgsRasterLayer, QgsCoordinateTransform, QgsCoordinateReferenceSystem,
    QgsPointXY, QgsRectangle, QgsProject, QgsMessageLog, Qgis,
    QgsNetworkAccessManager, QgsRasterPipe, QgsRasterProjector
)
from qgis.PyQt.QtNetwork import QNetworkRequest, QNetworkReply
from qgis.PyQt.QtCore import QUrl, QEventLoop

try:
    import cv2
    import requests
    from osgeo import gdal, osr
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


@dataclass
class GCP:
    """Ground Control Point"""
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float
    confidence: float = 1.0


@dataclass
class ProcessingParameters:
    """Parameters for georeferencing process"""
    raster_layer: QgsRasterLayer
    seed_lat: float
    seed_lon: float
    box_size: float
    output_dir: str
    target_crs: QgsCoordinateReferenceSystem
    zoom_level: int = 14
    use_edge_detection: bool = True
    use_clahe: bool = True
    manual_fallback: bool = True
    max_features: int = 5000  # Limit features for performance
    tile_cache_size: int = 50  # Cache downloaded tiles


class TileCache:
    """LRU cache for downloaded tiles to avoid re-downloading"""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get tile from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key].copy()
            return None
    
    def put(self, key: str, tile: np.ndarray):
        """Add tile to cache"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = tile.copy()
            self.access_order.append(key)


class GeoreferencingEngine(QObject):
    """Main georeferencing processing engine with performance optimizations"""
    
    # Signals
    progress_updated = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    processing_finished = pyqtSignal(bool, str)
    manual_mode_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.gcps = []
        self.reference_image = None
        self.target_image = None
        self.wmts_bounds = None
        self.tile_cache = TileCache()
        
        # Performance optimization settings
        self.max_image_size = 2048  # Resize large images for faster processing
        self.feature_detection_scale = 0.5  # Scale factor for feature detection
        self.parallel_downloads = 4  # Number of parallel tile downloads
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        if not DEPENDENCIES_AVAILABLE:
            self.error_occurred.emit(f"Required dependencies not available: {IMPORT_ERROR}")
            return False
        return True

    def process_georeferencing(self, params: ProcessingParameters):
        """Main processing function with optimizations"""
        try:
            if not self.check_dependencies():
                return
                
            self.log_message.emit("Starting optimized georeferencing process...")
            
            # Stage 1: Validate inputs
            self.progress_updated.emit(5, "Validating inputs...")
            if not self._validate_inputs(params):
                return
                
            # Stage 2: Build reference window (download WMTS tiles)
            self.progress_updated.emit(15, "Downloading reference tiles...")
            if not self._build_reference_window(params):
                return
                
            # Stage 3: Preprocess images with optimizations
            self.progress_updated.emit(35, "Preprocessing images...")
            if not self._preprocess_images_optimized(params):
                return
                
            # Stage 4: Feature extraction and matching with optimizations
            self.progress_updated.emit(55, "Extracting and matching features...")
            if not self._extract_and_match_features_optimized(params):
                if params.manual_fallback:
                    self.log_message.emit("Automatic matching failed. Switching to manual mode...")
                    self.manual_mode_requested.emit()
                    return
                else:
                    self.error_occurred.emit("Automatic matching failed and manual fallback is disabled")
                    return
                    
            # Stage 5: Generate GCPs
            self.progress_updated.emit(75, "Generating Ground Control Points...")
            if not self._generate_gcps(params):
                return
                
            # Stage 6: Warp and save
            self.progress_updated.emit(90, "Warping and saving georeferenced image...")
            output_path = self._warp_and_save_optimized(params)
            if not output_path:
                return
                
            self.progress_updated.emit(100, "Process completed successfully!")
            self.processing_finished.emit(True, output_path)
            
        except Exception as e:
            error_msg = f"Unexpected error during processing: {str(e)}"
            self.error_occurred.emit(error_msg)
            QgsMessageLog.logMessage(error_msg, "AutoGeoreferencer", Qgis.Critical)

    def _validate_inputs(self, params: ProcessingParameters) -> bool:
        """Validate input parameters"""
        if not params.raster_layer or not params.raster_layer.isValid():
            self.error_occurred.emit("Invalid raster layer selected")
            return False
            
        if not os.path.exists(params.output_dir):
            self.error_occurred.emit("Output directory does not exist")
            return False
            
        if not (-90 <= params.seed_lat <= 90):
            self.error_occurred.emit("Invalid latitude value")
            return False
            
        if not (-180 <= params.seed_lon <= 180):
            self.error_occurred.emit("Invalid longitude value")
            return False
            
        return True

    def _build_reference_window(self, params: ProcessingParameters) -> bool:
        """Download and mosaic WMTS tiles with parallel downloading"""
        try:
            # Calculate bounding box
            min_lat = params.seed_lat - params.box_size
            max_lat = params.seed_lat + params.box_size
            min_lon = params.seed_lon - params.box_size
            max_lon = params.seed_lon + params.box_size

            # Clamp to valid lat/lon ranges to avoid excessive tile requests
            min_lat = max(min_lat, -85.05112878)  # Web Mercator limit
            max_lat = min(max_lat, 85.05112878)
            min_lon = max(min_lon, -180.0)
            max_lon = min(max_lon, 180.0)
            
            self.log_message.emit(f"Reference window: {min_lat:.6f}, {min_lon:.6f} to {max_lat:.6f}, {max_lon:.6f}")
            
            # Convert to Web Mercator (EPSG:3857) for tile calculations
            source_crs = QgsCoordinateReferenceSystem("EPSG:4269")
            target_crs = QgsCoordinateReferenceSystem("EPSG:3857")
            transform = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())
            
            min_point = transform.transform(QgsPointXY(min_lon, min_lat))
            max_point = transform.transform(QgsPointXY(max_lon, max_lat))
            
            # Calculate tile bounds
            zoom = params.zoom_level
            tile_bounds = self._calculate_tile_bounds(min_point.x(), min_point.y(), 
                                                    max_point.x(), max_point.y(), zoom)
            
            # Download tiles with parallel processing
            tiles = self._download_wmts_tiles_parallel(tile_bounds, zoom, params.tile_cache_size)
            if not tiles:
                self.error_occurred.emit("Failed to download WMTS tiles")
                return False
                
            # Mosaic tiles
            self.reference_image = self._mosaic_tiles(tiles, tile_bounds, zoom)
            if self.reference_image is None:
                self.error_occurred.emit("Failed to create tile mosaic")
                return False
                
            # Store bounds for GCP generation
            self.wmts_bounds = {
                'min_x': min_point.x(),
                'min_y': min_point.y(),
                'max_x': max_point.x(),
                'max_y': max_point.y(),
                'zoom': zoom
            }
            
            self.log_message.emit(f"Successfully downloaded and mosaicked {len(tiles)} tiles")
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error building reference window: {str(e)}")
            return False

    @lru_cache(maxsize=32)
    def _calculate_tile_bounds(self, min_x: float, min_y: float, max_x: float, max_y: float, zoom: int) -> Dict:
        """Calculate tile bounds for given extent and zoom level (cached)"""
        # Web Mercator bounds
        EARTH_CIRCUMFERENCE = 40075016.686
        ORIGIN_SHIFT = EARTH_CIRCUMFERENCE / 2.0
        
        # Convert to tile coordinates
        tile_size = EARTH_CIRCUMFERENCE / (2 ** zoom)
        
        min_tile_x = int((min_x + ORIGIN_SHIFT) / tile_size)
        max_tile_x = int((max_x + ORIGIN_SHIFT) / tile_size)
        min_tile_y = int((ORIGIN_SHIFT - max_y) / tile_size)
        max_tile_y = int((ORIGIN_SHIFT - min_y) / tile_size)

        # Clamp to valid tile ranges
        max_tile_index = (2 ** zoom) - 1
        min_tile_x = max(0, min(max_tile_index, min_tile_x))
        max_tile_x = max(0, min(max_tile_index, max_tile_x))
        min_tile_y = max(0, min(max_tile_index, min_tile_y))
        max_tile_y = max(0, min(max_tile_index, max_tile_y))
        
        return {
            'min_x': min_tile_x,
            'max_x': max_tile_x,
            'min_y': min_tile_y,
            'max_y': max_tile_y,
            'tile_size': tile_size
        }

    def _download_single_tile(self, x: int, y: int, zoom: int) -> Optional[Dict]:
        """Download a single tile with caching"""
        cache_key = f"{zoom}_{y}_{x}"
        
        # Check cache first
        cached_tile = self.tile_cache.get(cache_key)
        if cached_tile is not None:
            return {
                'image': cached_tile,
                'x': x,
                'y': y,
                'zoom': zoom
            }
        
        try:
            base_url = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile"
            url = f"{base_url}/{zoom}/{y}/{x}"
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Convert to OpenCV image
                img_array = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Cache the tile
                    self.tile_cache.put(cache_key, img)
                    
                    return {
                        'image': img,
                        'x': x,
                        'y': y,
                        'zoom': zoom
                    }
        except Exception as e:
            self.log_message.emit(f"Failed to download tile {x},{y}: {str(e)}")
        
        return None

    def _download_wmts_tiles_parallel(self, tile_bounds: Dict, zoom: int, cache_size: int) -> List[Dict]:
        """Download WMTS tiles using parallel processing"""
        tiles = []
        
        # Generate list of tile coordinates
        tile_coords = []
        for x in range(tile_bounds['min_x'], tile_bounds['max_x'] + 1):
            for y in range(tile_bounds['min_y'], tile_bounds['max_y'] + 1):
                tile_coords.append((x, y, zoom))
        
        total_tiles = len(tile_coords)
        downloaded = 0
        
        # Use ThreadPoolExecutor for parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_downloads) as executor:
            # Submit all download tasks
            future_to_coords = {
                executor.submit(self._download_single_tile, x, y, zoom): (x, y, zoom)
                for x, y, zoom in tile_coords
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_coords):
                tile_data = future.result()
                if tile_data:
                    tiles.append(tile_data)
                
                downloaded += 1
                progress = 15 + int((downloaded / total_tiles) * 20)  # 15-35% range
                self.progress_updated.emit(progress, f"Downloaded {downloaded}/{total_tiles} tiles")
        
        return tiles

    def _mosaic_tiles(self, tiles: List[Dict], tile_bounds: Dict, zoom: int) -> Optional[np.ndarray]:
        """Mosaic downloaded tiles into single image"""
        if not tiles:
            return None
            
        # Calculate mosaic dimensions
        cols = tile_bounds['max_x'] - tile_bounds['min_x'] + 1
        rows = tile_bounds['max_y'] - tile_bounds['min_y'] + 1
        
        # Assume 256x256 tile size (standard for most WMTS services)
        tile_size = 256
        mosaic_height = rows * tile_size
        mosaic_width = cols * tile_size
        
        # Create empty mosaic
        mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
        
        # Place tiles in mosaic
        for tile in tiles:
            col = tile['x'] - tile_bounds['min_x']
            row = tile['y'] - tile_bounds['min_y']
            
            start_y = row * tile_size
            end_y = start_y + tile_size
            start_x = col * tile_size
            end_x = start_x + tile_size
            
            # Ensure tile is correct size
            tile_img = tile['image']
            if tile_img.shape[:2] != (tile_size, tile_size):
                tile_img = cv2.resize(tile_img, (tile_size, tile_size))
            
            mosaic[start_y:end_y, start_x:end_x] = tile_img
        
        return mosaic

    def _resize_for_processing(self, image: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
        """Resize image for faster processing while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        if max(height, width) <= max_size:
            return image, 1.0
        
        # Calculate scale factor
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized, scale

    def _preprocess_images_optimized(self, params: ProcessingParameters) -> bool:
        """Preprocess both reference and target images with optimizations"""
        try:
            # Read target raster with optimized block reading
            provider = params.raster_layer.dataProvider()
            extent = params.raster_layer.extent()
            width = params.raster_layer.width()
            height = params.raster_layer.height()
            
            # Optimize raster reading for large images
            if width * height > 4000000:  # > 4MP
                # Read at reduced resolution for initial processing
                scale_factor = math.sqrt(4000000 / (width * height))
                read_width = int(width * scale_factor)
                read_height = int(height * scale_factor)
                self.log_message.emit(f"Large image detected, reading at {read_width}x{read_height} for processing")
            else:
                read_width, read_height = width, height
            
            # Read raster data
            block = provider.block(1, extent, read_width, read_height)
            if not block.isValid():
                self.error_occurred.emit("Failed to read raster data")
                return False
            
            # Convert to numpy array with correct data type
            raw_bytes = block.data()
            expected_pixels = read_width * read_height
            bytes_per_pixel = max(1, len(raw_bytes) // expected_pixels)

            if bytes_per_pixel >= 8:
                dtype = np.float64
            elif bytes_per_pixel >= 4:
                dtype = np.float32
            elif bytes_per_pixel >= 2:
                dtype = np.uint16
            else:
                dtype = np.uint8

            data = np.frombuffer(raw_bytes, dtype=dtype).reshape((read_height, read_width)).astype(np.float32)
            
            # Normalize to 0-255 range
            data_min, data_max = np.nanmin(data), np.nanmax(data)
            if data_max > data_min:
                data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                data = np.zeros_like(data, dtype=np.uint8)
            
            # Resize for processing if needed
            self.target_image, self.target_scale = self._resize_for_processing(data, self.max_image_size)
            
            # Convert reference image to grayscale and resize
            if len(self.reference_image.shape) == 3:
                ref_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
            else:
                ref_gray = self.reference_image.copy()
            
            ref_gray, self.ref_scale = self._resize_for_processing(ref_gray, self.max_image_size)
            
            # Apply preprocessing with optimized parameters
            if params.use_clahe:
                # Use smaller tile size for faster processing
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                self.target_image = clahe.apply(self.target_image)
                ref_gray = clahe.apply(ref_gray)
            
            if params.use_edge_detection:
                # Use optimized Canny parameters
                self.target_image = cv2.Canny(self.target_image, 50, 150, apertureSize=3)
                ref_gray = cv2.Canny(ref_gray, 50, 150, apertureSize=3)
            
            self.reference_image = ref_gray
            
            self.log_message.emit("Optimized image preprocessing completed")
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error preprocessing images: {str(e)}")
            return False

    def _extract_and_match_features_optimized(self, params: ProcessingParameters) -> bool:
        """Extract features and find matches with performance optimizations"""
        try:
            # Initialize feature detector with optimized parameters
            try:
                # Use SIFT with limited features for performance
                detector = cv2.SIFT_create(nfeatures=params.max_features, contrastThreshold=0.04)
                self.log_message.emit(f"Using SIFT feature detector (max {params.max_features} features)")
            except AttributeError:
                # Fallback to ORB with optimized parameters
                detector = cv2.ORB_create(
                    nfeatures=params.max_features,
                    scaleFactor=1.2,
                    nlevels=8,
                    edgeThreshold=15,
                    firstLevel=0,
                    WTA_K=2,
                    scoreType=cv2.ORB_HARRIS_SCORE,
                    patchSize=31,
                    fastThreshold=20
                )
                self.log_message.emit(f"Using optimized ORB feature detector (max {params.max_features} features)")
            
            # Detect keypoints and descriptors with region of interest
            # Focus on central region for better matches
            h, w = self.target_image.shape[:2]
            roi_margin = 0.1  # 10% margin
            roi = (
                int(w * roi_margin), int(h * roi_margin),
                int(w * (1 - roi_margin)), int(h * (1 - roi_margin))
            )
            
            # Create masks for region of interest
            target_mask = np.zeros(self.target_image.shape[:2], dtype=np.uint8)
            target_mask[roi[1]:roi[3], roi[0]:roi[2]] = 255
            
            ref_h, ref_w = self.reference_image.shape[:2]
            ref_mask = np.zeros(self.reference_image.shape[:2], dtype=np.uint8)
            ref_mask[int(ref_h * roi_margin):int(ref_h * (1 - roi_margin)),
                     int(ref_w * roi_margin):int(ref_w * (1 - roi_margin))] = 255
            
            kp1, desc1 = detector.detectAndCompute(self.target_image, target_mask)
            kp2, desc2 = detector.detectAndCompute(self.reference_image, ref_mask)
            
            if desc1 is None or desc2 is None:
                self.log_message.emit("No features detected in one or both images")
                return False
            
            self.log_message.emit(f"Detected {len(kp1)} features in target, {len(kp2)} in reference")
            
            # Match features with optimized matcher
            if desc1.dtype == np.float32:
                # Use FLANN matcher for SIFT
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
                
                # Use knnMatch for better filtering
                matches = matcher.knnMatch(desc1, desc2, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                matches = good_matches
            else:
                # Use BFMatcher for ORB
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(desc1, desc2)
            
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) < 10:
                self.log_message.emit(f"Insufficient matches found: {len(matches)}")
                return False
            
            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Find homography using RANSAC with optimized parameters
            homography, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=2000,
                confidence=0.99
            )
            
            if homography is None:
                self.log_message.emit("Failed to compute homography")
                return False
            
            # Store inlier matches for GCP generation
            inliers = mask.ravel() == 1
            self.matched_points = {
                'src': src_pts[inliers],
                'dst': dst_pts[inliers],
                'homography': homography
            }
            
            inlier_count = np.sum(inliers)
            self.log_message.emit(f"Found {inlier_count} inlier matches out of {len(matches)} total matches")
            
            return inlier_count >= 6  # Need at least 6 points for reliable georeferencing
            
        except Exception as e:
            self.error_occurred.emit(f"Error in feature matching: {str(e)}")
            return False

    def _generate_gcps(self, params: ProcessingParameters) -> bool:
        """Generate Ground Control Points from matched features"""
        try:
            self.gcps = []
            
            # Get image-to-world transform parameters for reference image
            bounds = self.wmts_bounds
            ref_height, ref_width = self.reference_image.shape[:2]
            
            # Account for scaling during preprocessing
            ref_height_orig = ref_height / self.ref_scale
            ref_width_orig = ref_width / self.ref_scale
            
            # Calculate pixel-to-world scaling
            world_width = bounds['max_x'] - bounds['min_x']
            world_height = bounds['max_y'] - bounds['min_y']
            
            pixel_to_world_x = world_width / ref_width_orig
            pixel_to_world_y = world_height / ref_height_orig
            
            # Generate GCPs from matched points
            src_points = self.matched_points['src'].reshape(-1, 2)
            dst_points = self.matched_points['dst'].reshape(-1, 2)
            
            # Account for scaling in coordinates
            src_points = src_points / self.target_scale
            dst_points = dst_points / self.ref_scale
            
            # Select well-distributed points (max 12 for stability)
            n_gcps = min(12, len(src_points))
            indices = self._select_distributed_points(src_points, n_gcps)
            
            for i in indices:
                # Target image pixel coordinates
                pixel_x = float(src_points[i][0])
                pixel_y = float(src_points[i][1])
                
                # Reference image pixel coordinates
                ref_pixel_x = float(dst_points[i][0])
                ref_pixel_y = float(dst_points[i][1])
                
                # Convert reference pixel to world coordinates
                world_x = bounds['min_x'] + ref_pixel_x * pixel_to_world_x
                world_y = bounds['max_y'] - ref_pixel_y * pixel_to_world_y  # Y axis is flipped
                
                # Transform from Web Mercator to target CRS
                source_crs = QgsCoordinateReferenceSystem("EPSG:3857")
                transform = QgsCoordinateTransform(source_crs, params.target_crs, QgsProject.instance())
                world_point = transform.transform(QgsPointXY(world_x, world_y))
                
                gcp = GCP(
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                    world_x=world_point.x(),
                    world_y=world_point.y(),
                    confidence=1.0
                )
                self.gcps.append(gcp)
            
            self.log_message.emit(f"Generated {len(self.gcps)} Ground Control Points")
            return len(self.gcps) >= 3
            
        except Exception as e:
            self.error_occurred.emit(f"Error generating GCPs: {str(e)}")
            return False

    def _select_distributed_points(self, points: np.ndarray, n_points: int) -> List[int]:
        """Select well-distributed points from the matched set using spatial distribution"""
        if len(points) <= n_points:
            return list(range(len(points)))
        
        # Use k-means clustering to select distributed points
        try:
            from sklearn.cluster import KMeans
            
            # Use k-means to find n_points clusters
            kmeans = KMeans(n_clusters=n_points, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(points)
            
            # Select the point closest to each cluster center
            selected = []
            for i in range(n_points):
                cluster_points = points[clusters == i]
                if len(cluster_points) > 0:
                    cluster_center = kmeans.cluster_centers_[i]
                    # Find closest point to cluster center
                    distances = np.sum((cluster_points - cluster_center) ** 2, axis=1)
                    closest_idx = np.argmin(distances)
                    # Get original index
                    original_indices = np.where(clusters == i)[0]
                    selected.append(original_indices[closest_idx])
            
            return selected
            
        except ImportError:
            # Fallback to grid-based selection if sklearn not available
            return self._select_distributed_points_grid(points, n_points)
    
    def _select_distributed_points_grid(self, points: np.ndarray, n_points: int) -> List[int]:
        """Fallback grid-based point selection"""
        selected = []
        
        # Get image bounds
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        
        # Create grid
        grid_size = int(np.ceil(np.sqrt(n_points)))
        cell_width = (max_x - min_x) / grid_size
        cell_height = (max_y - min_y) / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(selected) >= n_points:
                    break
                    
                # Define cell bounds
                cell_min_x = min_x + i * cell_width
                cell_max_x = min_x + (i + 1) * cell_width
                cell_min_y = min_y + j * cell_height
                cell_max_y = min_y + (j + 1) * cell_height
                
                # Find points in this cell
                in_cell = ((points[:, 0] >= cell_min_x) & (points[:, 0] < cell_max_x) &
                          (points[:, 1] >= cell_min_y) & (points[:, 1] < cell_max_y))
                
                cell_indices = np.where(in_cell)[0]
                if len(cell_indices) > 0:
                    # Select the first point in the cell (they're already sorted by match quality)
                    selected.append(cell_indices[0])
        
        # If we don't have enough points, add more from remaining points
        remaining = [i for i in range(len(points)) if i not in selected]
        while len(selected) < n_points and remaining:
            selected.append(remaining.pop(0))
        
        return selected[:n_points]

    def _warp_and_save_optimized(self, params: ProcessingParameters) -> Optional[str]:
        """Warp the image using GCPs and save with optimizations"""
        try:
            # Create output filename
            base_name = os.path.splitext(os.path.basename(params.raster_layer.source()))[0]
            output_filename = f"{base_name}_georeferenced.tif"
            output_path = os.path.join(params.output_dir, output_filename)
            
            # Open source dataset
            src_ds = gdal.Open(params.raster_layer.source())
            if not src_ds:
                self.error_occurred.emit("Failed to open source raster")
                return None
            
            # Create GCP list for GDAL
            gcp_list = []
            for i, gcp in enumerate(self.gcps):
                gdal_gcp = gdal.GCP(gcp.world_x, gcp.world_y, 0, gcp.pixel_x, gcp.pixel_y, f"GCP_{i}", str(i))
                gcp_list.append(gdal_gcp)
            
            # Create temporary VRT with GCPs
            vrt_path = output_path.replace('.tif', '_temp.vrt')
            vrt_ds = gdal.Translate(vrt_path, src_ds, GCPs=gcp_list)
            if not vrt_ds:
                self.error_occurred.emit("Failed to create VRT with GCPs")
                return None
            
            # Set target CRS
            target_srs = osr.SpatialReference()
            target_srs.ImportFromEPSG(int(params.target_crs.authid().split(':')[1]))
            vrt_ds.SetProjection(target_srs.ExportToWkt())
            vrt_ds = None  # Close VRT
            
            # Warp to final output with optimized settings
            warp_options = gdal.WarpOptions(
                dstSRS=target_srs.ExportToWkt(),
                resampleAlg=gdal.GRA_Cubic,
                errorThreshold=0.125,
                creationOptions=[
                    'COMPRESS=LZW',
                    'TILED=YES',
                    'BLOCKXSIZE=512',
                    'BLOCKYSIZE=512',
                    'BIGTIFF=IF_SAFER'
                ],
                multithread=True,  # Enable multithreading
                warpMemoryLimit=512  # Limit memory usage
            )
            
            result_ds = gdal.Warp(output_path, vrt_path, options=warp_options)
            if not result_ds:
                self.error_occurred.emit("Failed to warp image")
                return None
            
            result_ds = None  # Close dataset
            
            # Clean up temporary VRT
            if os.path.exists(vrt_path):
                os.remove(vrt_path)
            
            # Add to QGIS project
            layer_name = f"{base_name}_georeferenced"
            georef_layer = QgsRasterLayer(output_path, layer_name)
            if georef_layer.isValid():
                QgsProject.instance().addMapLayer(georef_layer)
                self.log_message.emit(f"Added georeferenced layer to project: {layer_name}")
            
            self.log_message.emit(f"Georeferenced image saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.error_occurred.emit(f"Error warping and saving: {str(e)}")
            return None
