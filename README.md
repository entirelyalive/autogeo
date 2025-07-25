# QGIS Auto Georeferencer Plugin

A QGIS plugin that automatically georeferences un-referenced raster images (scanned aerial photos) by matching them to a USGS topographic WMTS basemap using computer vision techniques.

## Features

- **Automatic Dependency Installation**: Automatically installs required Python packages on first run
- **Automatic Feature Matching**: Uses SIFT/ORB feature detection and matching algorithms
- **WMTS Integration**: Downloads and mosaics USGS topographic tiles as reference
- **Manual Fallback**: Graceful fallback to manual tie-point selection when automatic matching fails
- **Processing Integration**: Available both as GUI tool and Processing algorithm
- **Performance Optimizations**: Parallel downloads, tile caching, and optimized image processing
- **Flexible Parameters**: Configurable seed coordinates, box size, zoom levels, and preprocessing options

## Automatic Dependency Management

The plugin automatically handles installation of required dependencies:

### Required Dependencies
- `opencv-python` (cv2) - For computer vision operations
- `requests` - For downloading WMTS tiles  
- `numpy` - For array operations

### Installation Process
1. **First Run Check**: Plugin checks for missing dependencies on startup
2. **User Consent**: Prompts user to install missing packages automatically
3. **Background Installation**: Uses `pip install --user` in a separate thread
4. **Progress Feedback**: Shows installation progress with cancellation option
5. **Verification**: Confirms successful installation before proceeding

### Manual Installation
If automatic installation fails, you can install manually:
```bash
pip install opencv-python requests numpy
```

## Performance Optimizations

### Tile Management
- **LRU Cache**: Caches downloaded tiles to avoid re-downloading (configurable size)
- **Parallel Downloads**: Downloads multiple tiles simultaneously (4 threads by default)
- **Smart Caching**: Persistent cache across plugin sessions

### Image Processing
- **Adaptive Scaling**: Automatically resizes large images for faster processing
- **ROI Processing**: Focuses feature detection on central regions
- **Optimized Parameters**: Tuned SIFT/ORB parameters for better performance
- **Memory Management**: Limits memory usage during warping operations

### Feature Matching
- **Limited Features**: Configurable maximum features (default 5000) for performance
- **FLANN Matching**: Uses fast approximate nearest neighbor matching for SIFT
- **Ratio Testing**: Applies Lowe's ratio test for better match quality
- **Spatial Distribution**: Intelligent GCP selection for optimal georeferencing

## Usage

### GUI Interface
1. Load a raster layer into your QGIS project
2. Go to **Raster → Auto Georeferencer**
3. **Dependency Check**: Plugin will automatically check and install dependencies if needed
4. Configure parameters:
   - Select the raster layer to georeference
   - Set seed latitude/longitude (approximate center of your image)
   - Adjust half-box size (search area around seed point)
   - Choose output directory
   - Set target coordinate reference system
5. **Advanced Options** (optional):
   - Max features for detection
   - Tile cache size
   - Parallel download settings
6. Click **OK** to start processing

### Processing Toolbox
The plugin is also available in the Processing Toolbox under **Auto Georeferencer → Auto Georeference Raster**.

## Algorithm Overview

### Stage 1: Dependency & Input Validation
- Checks and installs required dependencies automatically
- Validates raster layer and parameters
- Checks coordinate bounds and output directory

### Stage 2: Reference Window Building (Optimized)
- Converts seed coordinates to Web Mercator (EPSG:3857)
- Calculates appropriate WMTS zoom level
- Downloads tiles in parallel with caching
- Mosaics tiles into reference image

### Stage 3: Image Preprocessing (Optimized)
- Adaptive image scaling for performance
- Converts images to grayscale
- Applies CLAHE contrast enhancement (optional)
- Performs edge detection using Canny filter (optional)

### Stage 4: Feature Extraction and Matching (Optimized)
- Detects keypoints using SIFT or ORB algorithms with ROI focus
- Matches features using FLANN or BFMatcher
- Applies ratio testing and RANSAC filtering
- Uses optimized parameters for speed and accuracy

### Stage 5: Ground Control Point Generation
- Converts matched pixel coordinates to world coordinates
- Selects well-distributed points using spatial clustering
- Transforms coordinates to target CRS
- Accounts for image scaling during processing

### Stage 6: Warping and Output (Optimized)
- Creates GCP-based VRT file
- Warps image using GDAL with multithreading
- Uses optimized creation options (tiling, compression)
- Saves georeferenced result and adds to project

## Configuration Options

### Basic Options
- **Use Edge Detection**: Applies Canny edge detection for better feature matching
- **Apply CLAHE**: Enhances contrast using Contrast Limited Adaptive Histogram Equalization
- **Manual Fallback**: Enables manual tie-point mode when automatic matching fails
- **WMTS Zoom Level**: Controls resolution of reference tiles (1-18)

### Advanced Options
- **Max Features**: Maximum number of features to detect (500-10000)
- **Tile Cache Size**: Number of tiles to cache in memory (10-200)
- **Parallel Downloads**: Enable/disable parallel tile downloading

## Manual Fallback Mode

When automatic matching fails, the plugin can switch to manual mode:
1. Displays both target and reference images side-by-side
2. User clicks corresponding points on both images
3. Minimum 3 tie points required for georeferencing
4. Interactive table shows all created tie points

## File Structure

```
auto_georeferencer/
├── __init__.py                          # Plugin initialization
├── metadata.txt                         # Plugin metadata
├── dependency_manager.py                # Automatic dependency installation
├── auto_georeferencer.py               # Main plugin class (enhanced)
├── auto_georeferencer_dialog.py        # Main dialog interface (enhanced)
├── auto_georeferencer_dialog_base.ui   # UI layout file (enhanced)
├── georeferencing_engine.py            # Core processing engine (optimized)
├── manual_georeferencing_dialog.py     # Manual fallback interface
├── manual_georeferencing_dialog_base.ui # Manual mode UI layout
├── processing_provider.py              # Processing provider
├── processing_algorithm.py             # Processing algorithm
├── resources.py                        # Resource definitions
├── icon.png                           # Plugin icon
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Technical Details

### Dependency Management
- **Thread-Safe Installation**: Uses QThread for non-blocking installation
- **User Consent**: Always asks permission before installing packages
- **Progress Tracking**: Shows detailed installation progress
- **Error Handling**: Graceful fallback with manual installation instructions
- **Settings Persistence**: Remembers user preferences and installation status

### Performance Enhancements
- **Tile Caching**: LRU cache with configurable size and thread safety
- **Parallel Processing**: Concurrent tile downloads and feature processing
- **Memory Optimization**: Adaptive image scaling and memory-limited warping
- **Smart Algorithms**: Optimized feature detection and matching parameters

### WMTS Service
- Uses USGS National Map topographic basemap
- Base URL: `https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile`
- Standard 256x256 pixel tiles in Web Mercator projection
- Intelligent caching to reduce server load

### Feature Detection
- Primary: SIFT (Scale-Invariant Feature Transform) with FLANN matching
- Fallback: ORB (Oriented FAST and Rotated BRIEF) with BFMatcher
- RANSAC outlier filtering with optimized parameters
- Spatial distribution analysis for GCP selection

### Coordinate Systems
- Input seed coordinates: EPSG:4269 (NAD83)
- WMTS tiles: EPSG:3857 (Web Mercator)
- Output: User-configurable CRS

## Troubleshooting

### Dependency Issues

1. **"Installation Failed"**
   - Check internet connection
   - Ensure QGIS has write permissions
   - Try manual installation: `pip install opencv-python requests numpy`
   - Restart QGIS after manual installation

2. **"Dependencies not available after installation"**
   - Restart QGIS to refresh Python environment
   - Check if packages were installed in correct Python environment
   - Use "Check Dependencies" button to verify installation

### Processing Issues

1. **"No features detected"**
   - Try different preprocessing options
   - Ensure images have sufficient contrast and detail
   - Consider manual fallback mode
   - Adjust max features setting in advanced options

2. **"Insufficient matches found"**
   - Adjust seed coordinates to better match image center
   - Increase box size to cover more area
   - Try different zoom levels
   - Enable edge detection and CLAHE enhancement

3. **"Failed to download WMTS tiles"**
   - Check internet connection
   - Verify seed coordinates are within valid range
   - Try different zoom level
   - Check if tile cache is full (clear and retry)

### Performance Tips

- Use appropriate zoom levels (12-16 typically work best)
- Enable parallel downloads for faster tile retrieval
- Adjust tile cache size based on available memory
- Use advanced options to fine-tune performance
- Smaller box sizes process faster but may miss features
- Edge detection helps with low-contrast images

## License

This plugin is released under the GNU General Public License v3.0.

## Contributing

Contributions are welcome! Please submit issues and pull requests to the project repository.

## Support

For support and questions, please use the project's issue tracker or QGIS community forums.
