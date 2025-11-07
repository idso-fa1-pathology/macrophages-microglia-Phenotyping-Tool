import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, segmentation, feature
from skimage.color import label2rgb
import os
from scipy import ndimage
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import cv2
from tqdm import tqdm
import math
import numpy as np

def save_unified_cell_report(results, output_dir):
    """
    Saves a unified CSV file containing all IBA1+ cells with their morphology classification,
    nucleus counts, and Qki expression metrics.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from detect_nuclei_inside_iba1_with_qki_tiled
    output_dir : str
        Directory to save the report
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with comprehensive cell data
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating unified cell report...")
    
    # Extract necessary data
    iba1_props = results['iba1_props']
    nuclei_properties = results['nuclei_properties']
    morphology_classes = results.get('morphology_classes', {})
    iba1_labels = results['iba1_labels']
    
    # Prepare comprehensive cell data list
    all_cells_data = []
    
    # Process each IBA1+ cell
    for prop in iba1_props:
        cell_label = prop.label
        
        # Basic cell properties
        y, x = prop.centroid
        area = prop.area
        perimeter = prop.perimeter
        solidity = prop.solidity
        eccentricity = prop.eccentricity
        
        # Get morphology classification if available
        morphology_class = 'unknown'
        morphology_metrics = {}
        if morphology_classes and cell_label in morphology_classes:
            morph_data = morphology_classes[cell_label]
            morphology_class = morph_data['classification']
            morphology_metrics = morph_data.get('metrics', {})
        
        # Find nuclei associated with this cell
        cell_nuclei = []
        cell_mask = iba1_labels == cell_label
        
        # Approximate cell radius for nuclei association
        cell_radius = np.sqrt(area / np.pi)
        
        for nucleus in nuclei_properties:
            # Calculate distance from nucleus to cell centroid
            dist = np.sqrt((nucleus['y'] - y)**2 + (nucleus['x'] - x)**2)
            
            # Check if nucleus is within this cell (using a reasonable distance threshold)
            if dist <= 1.5 * cell_radius:
                cell_nuclei.append(nucleus)
        
        # Calculate Qki metrics (if present in nuclei data)
        nuclei_count = len(cell_nuclei)
        
        # Initialize Qki metrics
        nuclear_qki_values = []
        cytoplasmic_qki_values = []
        nc_ratio_values = []
        
        # Extract Qki values if available
        if nuclei_count > 0 and 'nuclear_qki' in cell_nuclei[0]:
            nuclear_qki_values = [n['nuclear_qki'] for n in cell_nuclei if 'nuclear_qki' in n]
            cytoplasmic_qki_values = [n['cytoplasmic_qki'] for n in cell_nuclei if 'cytoplasmic_qki' in n]
            nc_ratio_values = [n['nc_ratio'] for n in cell_nuclei 
                              if 'nc_ratio' in n and n['nc_ratio'] != float('inf')]
        
        # Calculate average metrics
        avg_nuclear_qki = np.mean(nuclear_qki_values) if nuclear_qki_values else 0
        avg_cytoplasmic_qki = np.mean(cytoplasmic_qki_values) if cytoplasmic_qki_values else 0
        avg_nc_ratio = np.mean(nc_ratio_values) if nc_ratio_values else 0
        
        # Create detailed cell data dictionary
        cell_data = {
            'cell_id': cell_label,
            'cell_type': 'iba1',
            'centroid_x': x,
            'centroid_y': y,
            'area': area,
            'perimeter': perimeter,
            'solidity': solidity,
            'eccentricity': eccentricity,
            'morphology_class': morphology_class,
            'has_nuclei': nuclei_count > 0,
            'nuclei_count': nuclei_count,
            'avg_nuclear_qki': avg_nuclear_qki,
            'avg_cytoplasmic_qki': avg_cytoplasmic_qki,
            'avg_nc_ratio': avg_nc_ratio,
        }
        
        # Add morphology metrics if available
        if morphology_metrics:
            for key, value in morphology_metrics.items():
                if key != 'label':  # Skip label to avoid duplication
                    cell_data[f'morph_{key}'] = value
        
        # Add individual nucleus data if present
        for i, nucleus in enumerate(cell_nuclei[:5]):  # Limit to first 5 nuclei to keep CSV manageable
            prefix = f'nucleus_{i+1}_'
            if 'nuclear_qki' in nucleus:
                cell_data[f'{prefix}nuclear_qki'] = nucleus['nuclear_qki']
                cell_data[f'{prefix}cytoplasmic_qki'] = nucleus['cytoplasmic_qki']
                cell_data[f'{prefix}nc_ratio'] = nucleus['nc_ratio']
                cell_data[f'{prefix}x'] = nucleus['x']
                cell_data[f'{prefix}y'] = nucleus['y']
        
        # Add cell to the list
        all_cells_data.append(cell_data)
    
    # Create DataFrame and save to CSV
    if all_cells_data:
        df = pd.DataFrame(all_cells_data)
        csv_path = os.path.join(output_dir, "all_cells_comprehensive.csv")
        df.to_csv(csv_path, index=False)
        print(f"Unified cell report saved to: {csv_path}")
        
        # Also create separate reports for cell types
        iba1_cells = df[df['cell_type'] == 'iba1']
        
        # Save separate CSVs for IBA1 cells
        if not iba1_cells.empty:
            iba1_cells.to_csv(os.path.join(output_dir, "iba1_cells.csv"), index=False)
            
            # Further divide IBA1 cells by morphology and nuclei
            with_nuclei = iba1_cells[iba1_cells['nuclei_count'] > 0]
            without_nuclei = iba1_cells[iba1_cells['nuclei_count'] == 0]
            
            if not with_nuclei.empty:
                with_nuclei.to_csv(os.path.join(output_dir, "iba1_cells_with_nuclei.csv"), index=False)
            
            if not without_nuclei.empty:
                without_nuclei.to_csv(os.path.join(output_dir, "iba1_cells_without_nuclei.csv"), index=False)
                
            # Create separate files by morphology class
            for morph_class in iba1_cells['morphology_class'].unique():
                class_df = iba1_cells[iba1_cells['morphology_class'] == morph_class]
                if not class_df.empty:
                    class_df.to_csv(os.path.join(output_dir, f"iba1_{morph_class}_cells.csv"), index=False)
        
        # Save cell centers for spatial analysis
        centers_df = df[['cell_type', 'centroid_x', 'centroid_y']]
        centers_df.to_csv(os.path.join(output_dir, "cell_centers.csv"), index=False)
        
        # Save separate center files for IBA1 cells
        iba1_centers = centers_df[centers_df['cell_type'] == 'iba1']
        
        if not iba1_centers.empty:
            iba1_centers.to_csv(os.path.join(output_dir, "iba1_cell_centers.csv"), index=False)
            
        return df
    else:
        print("No cells found for unified report.")
        return None
def process_tile(args):
    """
    Ultra-optimized tile processing with all required keys in the output dictionary.
    """
    (tile_img, tile_coords, iba1_channel, dapi_channel, qki_channel,
     iba1_threshold_method, dapi_threshold_method, iba1_min_size, 
     iba1_max_size, dapi_min_size, dapi_max_size, cytoplasm_expansion) = args
    
    # Extract coordinates once
    x_start, y_start, x_end, y_end = tile_coords
    
    # Extract the channels we need
    dapi_img = tile_img[:, :, dapi_channel]
    qki_img = tile_img[:, :, qki_channel]
    
    # Create masks for all required keys
    # Use simple thresholding for placeholders
    iba1_mask = np.zeros_like(dapi_img, dtype=bool)
    if iba1_channel < tile_img.shape[2]:
        iba1_img = tile_img[:, :, iba1_channel]
        iba1_mask = iba1_img > np.percentile(iba1_img, 75)
    
    # Process DAPI nuclei
    if dapi_img.dtype != np.uint8:
        dapi_uint8 = cv2.normalize(dapi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        dapi_uint8 = dapi_img
    
    _, dapi_binary = cv2.threshold(dapi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dapi_binary = dapi_binary > 0
    
    # Simple cleanup
    min_size = max(10, dapi_min_size // 2)
    dapi_labels, num_labels = ndimage.label(dapi_binary)
    
    if num_labels > 0:
        component_sizes = np.bincount(dapi_labels.ravel())
        too_small = component_sizes < min_size
        too_small_mask = too_small[dapi_labels]
        dapi_binary[too_small_mask] = 0
    
    # Find nuclei in IBA1 regions
    nuclei_in_iba1 = dapi_binary & iba1_mask
    
    # Measure nuclei properties
    nuclei_props = []
    if np.any(nuclei_in_iba1):
        # Label nuclei in IBA1
        nuclei_labels, num_nuclei = ndimage.label(nuclei_in_iba1)
        
        # Process each nucleus
        for label in range(1, num_nuclei + 1):
            # Create mask for this nucleus
            nucleus_mask = nuclei_labels == label
            
            # Calculate area
            area = np.sum(nucleus_mask)
            
            # Skip if too small or too large
            if not (dapi_min_size <= area <= dapi_max_size):
                continue
            
            # Calculate centroid
            coords = np.where(nucleus_mask)
            y = coords[0].mean()
            x = coords[1].mean()
            
            # Get QKI intensities in nucleus
            nuclear_qki = np.mean(qki_img[nucleus_mask])
            
            # Create a dilated mask for cytoplasm using faster method
            nucleus_dilated = ndimage.binary_dilation(
                nucleus_mask, 
                structure=np.ones((3,3)), 
                iterations=1
            )
            
            # Get cytoplasm (dilated minus nucleus)
            cytoplasm = nucleus_dilated & ~nucleus_mask & iba1_mask
            
            # Measure QKI in cytoplasm
            if np.any(cytoplasm):
                cyto_qki = np.mean(qki_img[cytoplasm])
                nc_ratio = nuclear_qki / cyto_qki if cyto_qki > 0 else float('inf')
            else:
                cyto_qki = 0
                nc_ratio = float('inf')
            
            # Store only essential data
            nuclei_props.append({
                'x': x + x_start,
                'y': y + y_start,
                'area': area,
                'nuclear_qki': nuclear_qki,
                'cytoplasmic_qki': cyto_qki,
                'nc_ratio': nc_ratio
            })
    
    # Return ALL required keys
    return {
        'coords': tile_coords,
        'iba1_mask': iba1_mask,
        'nuclei_mask': dapi_binary,
        'nuclei_in_iba1_mask': nuclei_in_iba1,
        'nuclei_props': nuclei_props
    }

def enhance_iba1_detection(iba1_img, min_size=50, max_size=5000, 
                          remove_large_artifacts=True, artifact_threshold=10000,
                          min_intensity=None, max_intensity=None):
    """
    Advanced IBA1 cell detection with multi-scale processing and cell reconstruction
    to better handle various cell morphologies and intensities.
    
    Parameters:
    -----------
    iba1_img : ndarray
        Input IBA1 channel image
    min_size : int
        Minimum cell size to keep
    max_size : int
        Maximum cell size to keep
    remove_large_artifacts : bool
        Whether to perform additional filtering for large artifacts
    artifact_threshold : int
        Size threshold above which objects are considered potential artifacts
    min_intensity : float or None
        Minimum mean intensity threshold for keeping a cell. 
        Cells with mean intensity below this value will be filtered out.
        If None, no minimum intensity filtering is applied.
    max_intensity : float or None
        Maximum mean intensity threshold for keeping a cell.
        Cells with mean intensity above this value will be filtered out.
        If None, no maximum intensity filtering is applied.
        
    Returns:
    --------
    ndarray
        Binary mask of detected IBA1+ cells
    """
    # Normalize to 0-255 range
    iba1_normalized = cv2.normalize(iba1_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 1. Multi-scale processing to capture both cell bodies and processes
    
    # Scale 1: Focus on cell bodies (stronger denoising)
    iba1_bodies = cv2.GaussianBlur(iba1_normalized, (7, 7), 2)
    
    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    iba1_bodies_enhanced = clahe.apply(iba1_bodies)
    
    # Scale 2: Focus on fine processes (less denoising)
    iba1_processes = cv2.GaussianBlur(iba1_normalized, (3, 3), 0.5)
    iba1_processes_enhanced = clahe.apply(iba1_processes)
    
    # 2. Multi-level thresholding to capture different intensity structures
    
    # For cell bodies: use Otsu's threshold
    _, bodies_binary = cv2.threshold(
        iba1_bodies_enhanced, 
        0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # For processes: use a lower, more sensitive threshold (percentage of Otsu)
    _, otsu_binary = cv2.threshold(
        iba1_processes_enhanced, 
        0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Get the actual threshold value - it's returned differently than in the direct call
    otsu_thresh = filters.threshold_otsu(iba1_processes_enhanced)
    
    process_thresh = int(otsu_thresh * 0.6)  # 60% of Otsu threshold
    _, processes_binary = cv2.threshold(
        iba1_processes_enhanced, 
        process_thresh, 
        255, 
        cv2.THRESH_BINARY
    )
    
    # 3. Combine the two binary images
    combined_binary = cv2.bitwise_or(bodies_binary, processes_binary)
    
    # 4. Use adaptive thresholding for local variations
    block_size = max(31, int(min(iba1_img.shape) * 0.03))
    if block_size % 2 == 0:
        block_size += 1
        
    try:
        adaptive_binary = cv2.adaptiveThreshold(
            iba1_processes_enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            -3
        )
        # Convert to same datatype before combining
        adaptive_binary = adaptive_binary.astype(np.uint8)
        combined_binary = combined_binary.astype(np.uint8)
        # Combine with global threshold results
        combined_binary = cv2.bitwise_or(combined_binary, adaptive_binary)
    except Exception as e:
        print(f"Adaptive thresholding failed: {e}")
    
    # 5. Morphological operations to improve cell reconstruction
    
    # Convert to NumPy boolean for scikit-image operations
    binary_mask = combined_binary.astype(bool)
    
    # Close small gaps within cells (connect processes to cell body)
    binary_mask = morphology.closing(binary_mask, morphology.disk(2))
    
    # Remove small isolated pixels (noise)
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=20)
    
    # Fill small holes within cells
    binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=100)
    
    # 6. Conditional dilation to connect nearby cell fragments
    # Find potential cell centers
    distance = ndimage.distance_transform_edt(binary_mask)
    peaks = feature.peak_local_max(
        distance, 
        min_distance=15, 
        exclude_border=False,
        labels=binary_mask
    )
    
    # Create peak mask
    peak_mask = np.zeros_like(binary_mask, dtype=bool)
    for peak in peaks:
        peak_mask[peak[0], peak[1]] = True
    
    # Dilate peaks to reconnect fragmented cells
    dilated_peaks = morphology.dilation(peak_mask, morphology.disk(5))
    
    # Use dilated peaks as markers for reconstruction
    reconstructed = morphology.reconstruction(
        dilated_peaks & binary_mask,  # Intersection ensures we only reconstruct existing cell parts
        binary_mask,
        method='dilation'
    )
    
    # Convert to boolean since reconstruction might return float
    reconstructed_bool = reconstructed.astype(bool)
    
    # 7. Final cleanup
    # Remove small objects again
    final_mask = morphology.remove_small_objects(reconstructed_bool, min_size=min_size)
    
    # Apply watershed to separate touching cells
    distance = ndimage.distance_transform_edt(final_mask)
    
    # Find markers for watershed
    local_max_coords = feature.peak_local_max(
        distance, 
        min_distance=20,
        exclude_border=False,
        labels=final_mask
    )
    
    # Create marker mask
    local_max = np.zeros_like(distance, dtype=bool)
    for coord in local_max_coords:
        local_max[coord[0], coord[1]] = True
    
    markers = measure.label(local_max)
    
    # Apply watershed segmentation
    watershed_labels = segmentation.watershed(-distance, markers, mask=final_mask)
    
    # Filter by size and intensity
    props = measure.regionprops(watershed_labels, intensity_image=iba1_img)
    
    # Intensity-based filtering
    valid_labels = []
    for prop in props:
        # Basic size filter
        if not (min_size <= prop.area <= max_size):
            continue
            
        # Intensity-based filtering
        mean_intensity = prop.mean_intensity
        
        # Check minimum intensity threshold
        if min_intensity is not None and mean_intensity < min_intensity:
            continue
            
        # Check maximum intensity threshold
        if max_intensity is not None and mean_intensity > max_intensity:
            continue
            
        # For large objects, apply additional shape validation if enabled
        if remove_large_artifacts and prop.area > artifact_threshold:
            # Calculate shape metrics to identify artifacts
            solidity = prop.solidity  # Area / ConvexArea
            circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
            
            # Artifacts often have unusual shape characteristics
            # Real cells typically have reasonable solidity and aren't perfectly circular
            if solidity < 0.2 or solidity > 0.95:  # Too fragmented or too solid
                continue
            if circularity < 0.1 or circularity > 0.9:  # Too irregular or too perfect
                continue
            
            # Check for elongated structures that might be vessels or other artifacts
            if prop.major_axis_length > 0 and prop.minor_axis_length / prop.major_axis_length < 0.1:
                continue
        
        valid_labels.append(prop.label)
    
    # Create final mask with size-filtered and intensity-filtered regions
    final_mask = np.zeros_like(watershed_labels, dtype=bool)
    for label in valid_labels:
        final_mask[watershed_labels == label] = True
    
    return final_mask


def enhance_iba1_detection_in_tile(tile_img, iba1_channel, min_size=50, max_size=5000,
                            remove_large_artifacts=True, artifact_threshold=8000,
                            min_intensity=None, max_intensity=None):
    """
    Optimized IBA1 detection with faster processing
    """
    # Extract IBA1 channel as efficiently as possible
    iba1_img = tile_img[:, :, iba1_channel]
    
    # Convert to uint8 for faster CV operations
    iba1_uint8 = cv2.normalize(iba1_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply CLAHE with smaller tiles for faster processing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    iba1_enhanced = clahe.apply(iba1_uint8)
    
    # Use smaller blur kernel
    iba1_blurred = cv2.GaussianBlur(iba1_enhanced, (3, 3), 1)
    
    # Use simpler thresholding
    thresh_value = np.percentile(iba1_blurred, 75)
    iba1_binary = iba1_blurred > thresh_value
    
    # Optimize morphological operations
    # Use faster implementation with structured elements
    struct_elem = np.ones((3, 3), dtype=bool)
    iba1_binary = ndimage.binary_opening(iba1_binary, structure=struct_elem)
    iba1_binary = ndimage.binary_closing(iba1_binary, structure=struct_elem)
    
    # Remove small objects with optimized size threshold
    min_obj_size = max(25, min_size // 2)
    iba1_binary = morphology.remove_small_objects(iba1_binary, min_size=min_obj_size)
    
    # Label once and reuse
    iba1_labels = measure.label(iba1_binary)
    props = measure.regionprops(iba1_labels, intensity_image=iba1_img)
    
    # Filter with numpy operations
    valid_mask = np.zeros_like(iba1_binary, dtype=bool)
    
    for prop in props:
        # Basic size filter
        if not (min_size <= prop.area <= max_size):
            continue
            
        # Intensity filter if needed
        if min_intensity is not None and prop.mean_intensity < min_intensity:
            continue
            
        if max_intensity is not None and prop.mean_intensity > max_intensity:
            continue
            
        # Add this region to mask
        valid_mask[iba1_labels == prop.label] = True
    
    return valid_mask
def detect_nuclei_inside_iba1_with_qki_tiled(image_path, iba1_channel=1, dapi_channel=2, qki_channel=0,
                                     iba1_threshold_method="adaptive", dapi_threshold_method="otsu",
                                     iba1_min_size=50, iba1_max_size=5000,
                                     dapi_min_size=15, dapi_max_size=500,
                                     boundary_thickness=2, cytoplasm_expansion=3,
                                     tile_size=2000, overlap=50, max_workers=16,
                                     use_enhanced_iba1=True,
                                     remove_artifacts=True,
                                     artifact_area_threshold=8000,
                                     artifact_solidity_threshold=0.95,
                                     iba1_min_intensity=None,
                                     iba1_max_intensity=None
                                    ):
    """
    Optimized tiled implementation to detect DAPI+ nuclei inside IBA1+ cells and quantify Qki expression.
    
    Parameters:
    -----------
    image_path : str
        Path to the multi-channel image file
    iba1_channel, dapi_channel, qki_channel : int
        Channel indices for staining (0-indexed)
    iba1_threshold_method : str
        Methods for thresholding ('adaptive' is now a valid option)
    dapi_threshold_method : str
        Methods for thresholding DAPI
    iba1_min_size, iba1_max_size, dapi_min_size, dapi_max_size : int
        Size filters for cells and nuclei in pixels
    boundary_thickness : int
        Thickness of boundaries in visualization
    cytoplasm_expansion : int
        Number of pixels to expand nuclear mask to define cytoplasmic region
    tile_size : int
        Size of tiles to process (increased for better performance)
    overlap : int
        Overlap between adjacent tiles to handle boundary objects (reduced for better performance)
    max_workers : int
        Number of parallel workers to process tiles
    use_enhanced_iba1 : bool
        Whether to use the enhanced IBA1 detection algorithm
    remove_artifacts : bool
        Whether to apply additional artifact removal steps
    artifact_area_threshold : int
        Size threshold above which objects are considered potential artifacts
    artifact_solidity_threshold : float
        Solidity threshold for artifact identification
    iba1_min_intensity, iba1_max_intensity : float or None
        Intensity thresholds for IBA1+ cells
        
    Returns:
    --------
    dict
        Dictionary containing all processing results and measurements
    """
    start_time = time.time()
    print(f"Processing image: {image_path}")
    print(f"IBA1 intensity thresholds - Min: {iba1_min_intensity}, Max: {iba1_max_intensity}")
    print(f"Using 3 channels: QKI (ch{qki_channel}), IBA1 (ch{iba1_channel}), DAPI (ch{dapi_channel})")
    
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image with memory-efficient method
    try:
        # Try scikit-image first for multi-channel TIFF
        img = io.imread(image_path)
        print(f"Image loaded with scikit-image: shape={img.shape}")
        
        # If it's 3 channels and looks like BGR, convert to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Check if it looks like BGR (OpenCV default)
            # Usually channel 0 has more data than channel 2 in microscopy
            if np.mean(img[:,:,0]) < np.mean(img[:,:,2]):
                img = img[:, :, [2, 1, 0]]  # Convert BGR to RGB
                print("Converted BGR to RGB")
                
    except Exception as e:
        print(f"Error loading image: {e}")
        raise
    
    print(f"Image loaded: shape={img.shape}")
    
    # Validate channel indices
    num_channels = img.shape[2] if len(img.shape) > 2 else 1
    print(f"Image has {num_channels} channels")
    
    if max(qki_channel, iba1_channel, dapi_channel) >= num_channels:
        print(f"Warning: Some channel indices are out of bounds. The image has {num_channels} channels.")
    
    # Create tiles with optimized overlap
    height, width = img.shape[:2]
    tiles = []
    tile_coords = []
    
    # Use optimized tiling strategy (larger tiles, smaller overlap)
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            # Calculate tile boundaries with handling for image edges
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            # Extract tile only if it's large enough to process
            if (x_end - x > 100) and (y_end - y > 100):
                tile = img[y:y_end, x:x_end]
                tiles.append(tile)
                tile_coords.append((x, y, x_end, y_end))
    
    print(f"Image divided into {len(tiles)} tiles of approximate size {tile_size}x{tile_size} with {overlap}px overlap")
    
    # Results for enhanced detection
    enhanced_results = {}
    
    # OPTIMIZATION: Process enhanced IBA1 detection in parallel batches
    if use_enhanced_iba1:
        print("Using enhanced IBA1 detection algorithm with parallel processing...")
        
        # Pre-process tiles in parallel batches
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process in parallel using lambdas for simpler submission
            futures = [executor.submit(
                enhance_iba1_detection_in_tile,
                tile, 
                iba1_channel, 
                min_size=iba1_min_size, 
                max_size=iba1_max_size,
                remove_large_artifacts=remove_artifacts,
                artifact_threshold=artifact_area_threshold,
                min_intensity=iba1_min_intensity,
                max_intensity=iba1_max_intensity
            ) for tile in tiles]
            
            # Collect results (with progress output)
            enhanced_iba1_masks = []
            for i, future in enumerate(tqdm(futures, total=len(futures), desc="Processing IBA1 detection")):
                enhanced_iba1_masks.append(future.result())
        
        # Create a full resolution enhanced IBA1 mask
        full_enhanced_iba1 = np.zeros((height, width), dtype=bool)
        
        # Stitch the enhanced masks together more efficiently
        for i, mask in enumerate(enhanced_iba1_masks):
            if mask is not None:
                x_start, y_start, x_end, y_end = tile_coords[i]
                full_enhanced_iba1[y_start:y_end, x_start:x_end] |= mask
        
        # Store the enhanced mask for later use
        enhanced_results['enhanced_iba1_mask'] = full_enhanced_iba1
    
    # OPTIMIZATION: Update the process_tile arguments
    process_args = [(tiles[i], tile_coords[i], iba1_channel, dapi_channel, qki_channel,
                    iba1_threshold_method, dapi_threshold_method, iba1_min_size, 
                    iba1_max_size, dapi_min_size, dapi_max_size, cytoplasm_expansion) 
                   for i in range(len(tiles))]
    
    # Process tiles in parallel with optimized function and progress tracking
    print(f"Processing tiles with {max_workers} parallel workers...")
    results = []
    
    # Use a context manager to ensure resources are properly released
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and process with progress bar
        futures = [executor.submit(process_tile, arg) for arg in process_args]
        
        # Process results as they complete with progress tracking
        for future in tqdm(futures, total=len(futures), desc="Processing tiles"):
            result = future.result()
            if result is not None:  # Handle possible failures gracefully
                results.append(result)
    
    # Create full-size masks for results
    full_iba1_mask = np.zeros((height, width), dtype=bool)
    full_nuclei_mask = np.zeros((height, width), dtype=bool)
    full_nuclei_in_iba1_mask = np.zeros((height, width), dtype=bool)
    
    # Collect all properties
    all_nuclei_props = []
    
    # Stitch results together (with progress tracking)
    print("Stitching results from all tiles...")
    for result in tqdm(results, desc="Stitching tiles"):
        if 'coords' not in result:
            continue  # Skip invalid results
            
        x_start, y_start, x_end, y_end = result['coords']
        
        # Update masks using logical OR for overlapping regions (faster bitwise)
        mask_slice = (slice(y_start, y_end), slice(x_start, x_end))
        
        # Update using bitwise OR (much faster than looping)
        full_iba1_mask[mask_slice] |= result['iba1_mask']
        full_nuclei_mask[mask_slice] |= result['nuclei_mask']
        full_nuclei_in_iba1_mask[mask_slice] |= result['nuclei_in_iba1_mask']
        
        # Add nuclei properties
        if 'nuclei_props' in result:
            all_nuclei_props.extend(result['nuclei_props'])
    
    # If we have enhanced detection results, use them instead
    if 'enhanced_iba1_mask' in enhanced_results:
        print("Using enhanced IBA1 segmentation results...")
        full_iba1_mask = enhanced_results['enhanced_iba1_mask']
        
        # Recalculate nuclei in IBA1 using the enhanced mask (fast binary operation)
        full_nuclei_in_iba1_mask = full_nuclei_mask & full_iba1_mask
    
    # Additional post-processing to remove large artifacts if requested
    if remove_artifacts:
        print("Performing optimized artifact removal on full image...")
        
        # Process in chunks to reduce memory pressure (faster implementation)
        chunk_size = min(4000, height // 4)  # Adjust based on available memory
        filtered_iba1_mask = np.zeros_like(full_iba1_mask)
        
        # Process IBA1 cells in horizontal strips with progress tracking
        for y_start in tqdm(range(0, height, chunk_size), desc="IBA1 artifact removal"):
            y_end = min(y_start + chunk_size, height)
            
            # Extract chunk
            chunk_mask = full_iba1_mask[y_start:y_end, :]
            
            # Fast labeling with scikit-image
            chunk_labels = measure.label(chunk_mask)
            chunk_props = measure.regionprops(chunk_labels)
            
            # Filter artifacts with vectorized operations
            valid_labels = []
            for prop in chunk_props:
                # Skip very small objects
                if prop.area < 100:
                    valid_labels.append(prop.label)
                    continue
                    
                # For larger objects, check if they could be artifacts
                if prop.area > artifact_area_threshold:
                    # Calculate combined shape metric (faster than multiple checks)
                    solidity = prop.solidity
                    circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
                    elongation = prop.minor_axis_length / prop.major_axis_length if prop.major_axis_length > 0 else 1
                    
                    # Combined artifact score (higher = more likely artifact)
                    artifact_score = solidity * circularity * (1 - elongation)
                    
                    # Use threshold on combined score (faster decision)
                    if artifact_score > 0.7 or prop.area > 3 * artifact_area_threshold:
                        continue  # Skip this object (likely artifact)
                
                # If not an artifact, keep it
                valid_labels.append(prop.label)
            
            # Use fast numpy indexing to create valid mask
            valid_chunk_mask = np.isin(chunk_labels, valid_labels)
            
            # Update the filtered mask with this chunk
            filtered_iba1_mask[y_start:y_end, :] = valid_chunk_mask
        
        # Update the masks
        full_iba1_mask = filtered_iba1_mask
        
        # Recalculate nuclei in IBA1 using the filtered mask (fast binary operation)
        full_nuclei_in_iba1_mask = full_nuclei_mask & full_iba1_mask
    
    # Deduplicate nuclei that may appear in multiple tiles due to overlap
    # Use a spatial hash approach for much faster deduplication
    print("Deduplicating nuclei with spatial hashing...")
    deduplicated_nuclei = []
    spatial_hash = {}  # Hash map for fast lookup
    
    # Round coordinates to grid cells for faster comparison
    grid_size = 10  # Adjust based on expected cell density
    
    for nucleus in all_nuclei_props:
        # Create a spatial hash key
        coord_key = (int(nucleus['x'] / grid_size), int(nucleus['y'] / grid_size))
        
        # Check if we've already seen a nucleus in this grid cell
        if coord_key not in spatial_hash:
            spatial_hash[coord_key] = nucleus
            deduplicated_nuclei.append(nucleus)
        else:
            # If there's a collision, keep the one with better metrics
            # For example, higher QKI signal or better position
            existing = spatial_hash[coord_key]
            if nucleus.get('nuclear_qki', 0) > existing.get('nuclear_qki', 0):
                # Replace the existing nucleus with this one
                spatial_hash[coord_key] = nucleus
                # Find and replace in the deduplicated list
                for i, n in enumerate(deduplicated_nuclei):
                    if n is existing:
                        deduplicated_nuclei[i] = nucleus
                        break
    
    print(f"Deduplicated nuclei count: {len(deduplicated_nuclei)} (from original {len(all_nuclei_props)})")
    
    # Get the final labeled cells and nuclei (faster with skimage)
    print("Creating final labeled masks...")
    iba1_labels = measure.label(full_iba1_mask)
    nuclei_labels = measure.label(full_nuclei_mask)
    nuclei_in_iba1_labels = measure.label(full_nuclei_in_iba1_mask)
    
    # Analyze properties of the labeled objects (with progress tracking)
    print("Calculating region properties...")
    iba1_props = measure.regionprops(iba1_labels)
    nuclei_props = measure.regionprops(nuclei_labels)
    
    # ---------- STEP 4: CLASSIFY IBA1+ CELL MORPHOLOGY ----------
    # Add timing to track performance
    morph_start_time = time.time()
    print("Classifying cell morphology...")
    morphology_classes, morph_class_counts = classify_iba1_morphology(
        iba1_labels, 
        iba1_props,
        artifact_filter=remove_artifacts,
        artifact_solidity_threshold=artifact_solidity_threshold
    )
    morph_elapsed = time.time() - morph_start_time
    print(f"Morphology classification completed in {morph_elapsed:.2f} seconds")
    
    # Create morphology visualization (moved to separate visualization function)
    print("Creating morphology visualization...")
    morph_vis, morph_legend = visualize_iba1_morphology(iba1_labels, morphology_classes, 
                                                      (height, width))
    
    print(f"Morphology Classification Summary:")
    total_cells = sum(morph_class_counts.values())
    if total_cells > 0:
        for morph_class, count in morph_class_counts.items():
            if morph_class != 'artifact':  # Skip showing artifacts in calculation
                percentage = count/total_cells*100 if total_cells > 0 else 0
                print(f"{morph_class.capitalize()} cells: {count} ({percentage:.1f}%)")
        
        # Report artifacts separately if any were found
        if 'artifact' in morph_class_counts and morph_class_counts['artifact'] > 0:
            print(f"Potential artifacts filtered: {morph_class_counts['artifact']}")
    
    # Count the number of valid cells and nuclei
    iba1_count = len(iba1_props)
    nuclei_count = len(nuclei_props)
    nuclei_in_iba1_count = len(measure.regionprops(nuclei_in_iba1_labels))
    
    # Calculate statistics
    avg_nuclei_per_iba1 = nuclei_in_iba1_count / iba1_count if iba1_count > 0 else 0
    
    nuclear_qki_values = [nucleus['nuclear_qki'] for nucleus in deduplicated_nuclei]
    cytoplasmic_qki_values = [nucleus['cytoplasmic_qki'] for nucleus in deduplicated_nuclei]
    nc_ratio_values = [nucleus['nc_ratio'] for nucleus in deduplicated_nuclei 
                      if nucleus['nc_ratio'] != float('inf')]
    
    # Get background QKI level (5th percentile of image)
    qki_background = np.percentile(img[:, :, qki_channel], 5)
    
    # Calculate averages
    avg_nuclear_qki = np.mean(nuclear_qki_values) if nuclear_qki_values else 0
    avg_cyto_qki = np.mean(cytoplasmic_qki_values) if cytoplasmic_qki_values else 0
    avg_nc_ratio = np.mean(nc_ratio_values) if nc_ratio_values else 0
    
    # Prepare final results
    print(f"Analysis completed in {time.time() - start_time:.2f} seconds")
    print(f"Found {iba1_count} valid IBA1+ cells")
    print(f"Found {nuclei_count} valid DAPI+ nuclei")
    print(f"Found {nuclei_in_iba1_count} nuclei inside/overlapping with IBA1+ cells")
    print(f"Average nuclei per IBA1+ cell: {avg_nuclei_per_iba1:.2f}")
    print(f"Qki background level: {qki_background:.1f}")
    print(f"Average nuclear Qki intensity: {avg_nuclear_qki:.2f}")
    print(f"Average cytoplasmic Qki intensity: {avg_cyto_qki:.2f}")
    print(f"Average nuclear:cytoplasmic ratio: {avg_nc_ratio:.2f}")
    
    # Assemble final results dictionary (using minimal necessary data)
    results_dict = {
        'iba1_mask': full_iba1_mask,
        'nuclei_mask': full_nuclei_mask,
        'nuclei_in_iba1_mask': full_nuclei_in_iba1_mask,
        'iba1_labels': iba1_labels,
        'nuclei_labels': nuclei_labels,
        'nuclei_in_iba1_labels': nuclei_in_iba1_labels,
        'nuclei_properties': deduplicated_nuclei,
        'iba1_props': iba1_props,
        'nuclei_props': nuclei_props,
        'morphology_classes': morphology_classes,
        'morphology_vis': morph_vis,
        'morphology_legend': morph_legend,
        'stats': {
            'iba1_count': iba1_count,
            'nuclei_count': nuclei_count,
            'nuclei_in_iba1_count': nuclei_in_iba1_count,
            'avg_nuclei_per_iba1': avg_nuclei_per_iba1,
            'qki_background': qki_background,
            'avg_nuclear_qki': avg_nuclear_qki,
            'avg_cyto_qki': avg_cyto_qki,
            'avg_nc_ratio': avg_nc_ratio,
            'processing_time': time.time() - start_time,
            'morphology_counts': morph_class_counts,
            'artifacts_removed': morph_class_counts.get('artifact', 0)
        },
        'image': img,  # Keep original image for visualizations
        'artifact_filtering': {
            'enabled': remove_artifacts,
            'area_threshold': artifact_area_threshold,
            'solidity_threshold': artifact_solidity_threshold
        },
        'intensity_filtering': {
            'iba1_min_intensity': iba1_min_intensity,
            'iba1_max_intensity': iba1_max_intensity
        }
    }
    
    return results_dict

# Helper function for parallel processing
def process_iba1_batch(batch_tiles, batch_coords, iba1_channel, min_size, max_size, 
                      remove_artifacts, artifact_threshold, min_intensity, max_intensity):
    """Process a batch of tiles for IBA1 detection in parallel"""
    masks = []
    for i, tile in enumerate(batch_tiles):
        mask = enhance_iba1_detection_in_tile(
            tile, 
            iba1_channel, 
            min_size=min_size, 
            max_size=max_size,
            remove_large_artifacts=remove_artifacts,
            artifact_threshold=artifact_threshold,
            min_intensity=min_intensity,
            max_intensity=max_intensity
        )
        masks.append(mask)
    return masks
def create_visualizations(results, boundary_thickness=2):
    """
    Create visualization images from the analysis results at full resolution
    with corrected channel-to-color mapping
    """
    print("Creating visualizations at full resolution...")
    
    # Extract data from results
    img = results['image']
    iba1_mask = results['iba1_mask']
    nuclei_mask = results['nuclei_mask']
    nuclei_in_iba1_mask = results['nuclei_in_iba1_mask']
    iba1_labels = results['iba1_labels']
    nuclei_in_iba1_labels = results['nuclei_in_iba1_labels']
    morphology_vis = results.get('morphology_vis', None)
    
    # Create a composite visualization at full resolution
    composite = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    # Color mapping for 3 channels:
    # DAPI (channel 0) -> Blue
    # IBA1 (channel 1) -> Green
    # QKI (channel 2) -> Red
    
    # Extract channel indices from results
    dapi_channel = 0
    iba1_channel = 1
    qki_channel = 2
    
    # DAPI -> Blue channel
    channel = img[:, :, dapi_channel]
    p2, p98 = np.percentile(channel, (2, 98))
    normalized = np.clip((channel - p2) / (p98 - p2), 0, 1).astype(np.float32)
    composite[:, :, 2] = (normalized * 255).astype(np.uint8)  # Blue component
    
    # IBA1 -> Green channel
    channel = img[:, :, iba1_channel]
    p2, p98 = np.percentile(channel, (2, 98))
    normalized = np.clip((channel - p2) / (p98 - p2), 0, 1).astype(np.float32)
    composite[:, :, 1] = (normalized * 255).astype(np.uint8)  # Green component
    
    # QKI -> Red channel
    channel = img[:, :, qki_channel]
    p2, p98 = np.percentile(channel, (2, 98))
    normalized = np.clip((channel - p2) / (p98 - p2), 0, 1).astype(np.float32)
    composite[:, :, 0] = (normalized * 255).astype(np.uint8)  # Red component
    
    # Add IBA1 boundaries in cyan (green + blue)
    iba1_boundaries = segmentation.find_boundaries(iba1_mask, mode='outer', background=0)
    iba1_boundaries = morphology.binary_dilation(iba1_boundaries, morphology.disk(boundary_thickness))
    composite[iba1_boundaries, 0] = 0        # Red channel = 0
    composite[iba1_boundaries, 1] = 255      # Green channel = 255
    composite[iba1_boundaries, 2] = 255      # Blue channel = 255
    
    # Add nuclei in IBA1 boundaries in magenta
    nuclei_in_iba1_boundaries = segmentation.find_boundaries(nuclei_in_iba1_mask, mode='outer', background=0)
    nuclei_in_iba1_boundaries = morphology.binary_dilation(nuclei_in_iba1_boundaries, morphology.disk(boundary_thickness))
    composite[nuclei_in_iba1_boundaries, 0] = 255  # Red channel = 255
    composite[nuclei_in_iba1_boundaries, 1] = 0    # Green channel = 0
    composite[nuclei_in_iba1_boundaries, 2] = 255  # Blue channel = 255
    
    # Create labeled image - directly use full resolution
    labels_rgb = label2rgb(nuclei_in_iba1_labels, bg_label=0, bg_color=(0, 0, 0))
    
    # Create IBA1 overlay - directly use full resolution
    iba1_overlay = label2rgb(iba1_labels, bg_label=0, bg_color=(0, 0, 0))
    
    # Create Qki visualization if we have data for it
    qki_vis = None
    if img.shape[2] >= 3:  # We need at least 3 channels
        # Extract Qki channel
        qki_img = img[:, :, qki_channel]
        
        # Enhance contrast for visualization
        p2, p98 = np.percentile(qki_img, (2, 98))
        qki_enhanced = np.clip((qki_img - p2) / (p98 - p2), 0, 1).astype(np.float32)
        
        # Create Qki visualization at full resolution
        qki_vis = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
        qki_vis[:,:,0] = qki_enhanced  # Red channel for Qki
        qki_vis[:,:,1][iba1_boundaries] = 1.0  # Green boundaries for IBA1
        qki_vis[:,:,2][nuclei_in_iba1_boundaries] = 1.0  # Blue boundaries for nuclei in IBA1
    
    visualizations = {
        'composite': composite,
        'nuclei_in_iba1_labeled': labels_rgb,
        'iba1_labeled': iba1_overlay
    }
    
    # Add morphology and Qki visualizations if available
    if morphology_vis is not None:
        visualizations['morphology'] = morphology_vis
    
    if qki_vis is not None:
        visualizations['qki'] = (qki_vis * 255).astype(np.uint8)
    
    return visualizations


def classify_iba1_morphology(filtered_iba1_cells, iba1_props, 
                         artifact_filter=True, artifact_solidity_threshold=0.95):
    """
    Classifies IBA1+ cells into morphological categories with improved artifact filtering.
    
    Parameters:
    -----------
    filtered_iba1_cells : ndarray
        Labeled image of filtered IBA1+ cells
    iba1_props : list
        List of region properties for IBA1+ cells from skimage.measure.regionprops
    artifact_filter : bool
        Whether to apply additional filtering to remove artifacts
    artifact_solidity_threshold : float
        Solidity threshold above which cells might be considered artifacts
        
    Returns:
    --------
    dict
        Dictionary mapping cell labels to morphology classifications
    dict
        Dictionary with classification counts
    """
    # Dictionary to store classifications
    morphology_classes = {}
    
    # Count for each class
    class_counts = {
        'ameboid': 0,
        'rod': 0,
        'ramified': 0,
        'hyperramified': 0,
        'artifact': 0  # New category for artifacts
    }
    
    # Batch process metrics calculation for better performance
    print("Extracting cell metrics in batch...")
    # Process cells in chunks to avoid memory issues
    chunk_size = 1000
    all_metrics = []
    
    # Convert to numpy array once for faster operations
    filtered_iba1_np = np.asarray(filtered_iba1_cells)
    
    # Create a label set for faster lookups
    valid_labels = set(np.unique(filtered_iba1_np)) - {0}
    
    # Create progress reporting
    total_props = len(iba1_props)
    print(f"Processing {total_props} cells for morphology classification")
    
    # Process in batches
    for batch_start in range(0, total_props, chunk_size):
        batch_end = min(batch_start + chunk_size, total_props)
        batch_props = iba1_props[batch_start:batch_end]
        
        if batch_start % 5000 == 0:
            print(f"Processing batch {batch_start}-{batch_end} of {total_props}")
        
        batch_metrics = []
        for prop in batch_props:
            if prop.label == 0 or prop.label not in valid_labels:  # Skip background/invalid cells
                continue
                
            # Extract basic properties
            area = prop.area
            perimeter = prop.perimeter
            solidity = prop.solidity  # Ratio of cell area to convex hull area
            
            # Calculate circularity: 4π × area / perimeter²
            circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Calculate complexity: perimeter / sqrt(area)
            complexity = perimeter / math.sqrt(area) if area > 0 else 0
            
            # Get eccentricity
            eccentricity = prop.eccentricity
            
            # Calculate roundness: inverse of aspect ratio (avoid expensive convex hull)
            minor_axis = prop.minor_axis_length
            major_axis = prop.major_axis_length
            roundness = minor_axis / major_axis if major_axis > 0 else 0
            
            # Calculate additional metrics for artifact detection
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else float('inf')
            extent = area / (prop.bbox[2] - prop.bbox[0]) / (prop.bbox[3] - prop.bbox[1]) if (prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1]) > 0 else 0
            
            # Store metrics for this cell
            metrics = {
                'label': prop.label,
                'area': area,
                'perimeter': perimeter,
                'solidity': solidity,
                'circularity': circularity,
                'complexity': complexity,
                'eccentricity': eccentricity,
                'roundness': roundness,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'centroid_y': prop.centroid[0],
                'centroid_x': prop.centroid[1]
            }
            
            batch_metrics.append(metrics)
        
        # Add batch to all metrics
        all_metrics.extend(batch_metrics)
    
    # Skip morphology classification if there are too few cells for statistics
    if len(all_metrics) < 3:
        print("Too few cells for morphology classification")
        for metrics in all_metrics:
            morphology_classes[metrics['label']] = {
                'classification': 'unclassified',
                'metrics': metrics
            }
        class_counts['unclassified'] = len(all_metrics)
        return morphology_classes, class_counts
    
    # Use numpy for faster statistics calculation
    print("Calculating classification thresholds...")
    metrics_df = pd.DataFrame(all_metrics)
    
    # Calculate statistics using pandas for better performance
    stats = {}
    for metric in ['solidity', 'circularity', 'complexity', 'eccentricity', 'roundness', 'area', 'aspect_ratio', 'extent']:
        stats[metric] = {
            'p25': metrics_df[metric].quantile(0.25),
            'p50': metrics_df[metric].quantile(0.50),
            'p75': metrics_df[metric].quantile(0.75),
            'mean': metrics_df[metric].mean(),
            'std': metrics_df[metric].std()
        }
    
    # Print statistics
    print("\nMorphology Metrics Statistics:")
    for metric, values in stats.items():
        print(f"{metric}: 25th={values['p25']:.3f}, Median={values['p50']:.3f}, 75th={values['p75']:.3f}")
    
    # Set thresholds based on the distribution
    solidity_threshold_high = stats['solidity']['p75']
    solidity_threshold_mid = stats['solidity']['p50']
    solidity_threshold_low = stats['solidity']['p25']
    
    complexity_threshold_high = stats['complexity']['p75']
    complexity_threshold_low = metrics_df['complexity'].quantile(0.30)
    
    circularity_threshold = metrics_df['circularity'].quantile(0.70)
    
    eccentricity_threshold_high = stats['eccentricity']['p75']
    
    # Set artifact detection thresholds
    area_outlier_threshold = stats['area']['mean'] + 3 * stats['area']['std']  # 3 standard deviations above mean
    solidity_artifact_threshold = 0.95 if artifact_solidity_threshold > 1.0 else artifact_solidity_threshold
    roundness_artifact_threshold = 0.9  # Very round objects are suspicious
    aspect_ratio_artifact_threshold = 10  # Very elongated objects might be vessels
    
    # Identify potential artifacts first
    potential_artifacts = []
    if artifact_filter:
        for metrics in all_metrics:
            is_artifact = False
            
            # Check for size outliers (unusually large objects)
            if metrics['area'] > area_outlier_threshold:
                is_artifact = True
            
            # Check for shape characteristics typical of artifacts
            if metrics['solidity'] > solidity_artifact_threshold and metrics['area'] > stats['area']['p75']:
                # Very solid and large objects are often artifacts
                is_artifact = True
                
            # Unusually round large objects
            if metrics['roundness'] > roundness_artifact_threshold and metrics['circularity'] > 0.85 and metrics['area'] > stats['area']['p75']:
                is_artifact = True
                
            # Extremely elongated objects (potential vessels)
            if metrics['aspect_ratio'] > aspect_ratio_artifact_threshold:
                is_artifact = True
                
            if is_artifact:
                potential_artifacts.append(metrics['label'])
    
    # Classify cells in batches
    print("Classifying cells in batches...")
    for batch_start in range(0, len(all_metrics), chunk_size):
        batch_end = min(batch_start + chunk_size, len(all_metrics))
        batch_metrics = all_metrics[batch_start:batch_end]
        
        if batch_start % 5000 == 0:
            print(f"Classifying batch {batch_start}-{batch_end} of {len(all_metrics)}")
        
        for metrics in batch_metrics:
            # Check if this was identified as an artifact
            if metrics['label'] in potential_artifacts:
                classification = 'artifact'
                class_counts['artifact'] += 1
                morphology_classes[metrics['label']] = {
                    'classification': classification,
                    'metrics': metrics
                }
                continue
            
            # For non-artifacts, proceed with normal classification
            solidity = metrics['solidity']
            circularity = metrics['circularity']
            complexity = metrics['complexity']
            eccentricity = metrics['eccentricity']
            roundness = metrics['roundness']
            
            # Classification with simplified logic for better performance
            if solidity > solidity_threshold_high and circularity > circularity_threshold and complexity < complexity_threshold_low:
                classification = 'ameboid'
                class_counts['ameboid'] += 1
                
            elif (solidity_threshold_mid < solidity < solidity_threshold_high and 
                  eccentricity > eccentricity_threshold_high and 
                  complexity < (complexity_threshold_low + complexity_threshold_high)/2 and
                  roundness < 0.5):
                classification = 'rod'
                class_counts['rod'] += 1
                
            elif complexity > complexity_threshold_high and solidity < solidity_threshold_high and eccentricity > 0.7:
                classification = 'hyperramified'
                class_counts['hyperramified'] += 1
                
            elif solidity < solidity_threshold_low and complexity_threshold_low <= complexity <= complexity_threshold_high:
                classification = 'ramified'
                class_counts['ramified'] += 1
                
            else:
                # Simplified scoring for ambiguous cases
                ameboid_score = (solidity / solidity_threshold_high) + (circularity / circularity_threshold) + (1 - complexity/complexity_threshold_high)
                rod_score = (solidity / solidity_threshold_mid) + (eccentricity / eccentricity_threshold_high) + (1 - roundness) + (1 - complexity/(complexity_threshold_low + complexity_threshold_high)/2)
                ramified_score = (1 - solidity/solidity_threshold_high) + (complexity / complexity_threshold_high) + eccentricity
                hyperramified_score = (complexity / complexity_threshold_high) + (1 - solidity/solidity_threshold_high) + eccentricity
                
                scores = {
                    'ameboid': ameboid_score,
                    'rod': rod_score,
                    'ramified': ramified_score,
                    'hyperramified': hyperramified_score
                }
                
                classification = max(scores, key=scores.get)
                class_counts[classification] += 1
            
            # Store classification
            morphology_classes[metrics['label']] = {
                'classification': classification,
                'metrics': metrics
            }
    
    # Print classification results
    print(f"\nMorphology Classification Results:")
    for class_name, count in class_counts.items():
        print(f"{class_name.capitalize()} cells: {count}")
    
    # Print the thresholds
    print(f"\nClassification Thresholds:")
    print(f"Solidity: High > {solidity_threshold_high:.3f}, Mid > {solidity_threshold_mid:.3f}, Low < {solidity_threshold_low:.3f}")
    print(f"Complexity: High > {complexity_threshold_high:.3f}, Low < {complexity_threshold_low:.3f}")
    print(f"Circularity threshold: {circularity_threshold:.3f}")
    print(f"Eccentricity high threshold: {eccentricity_threshold_high:.3f}")
    
    if artifact_filter:
        print(f"Artifact filtering: Detected {len(potential_artifacts)} potential artifacts")
    
    return morphology_classes, class_counts


def visualize_iba1_morphology(filtered_iba1_cells, morphology_classes, img_shape):
    """
    Creates a visualization of IBA1+ cell morphology classification with optimized performance
    and handling for potential artifacts
    
    Parameters:
    -----------
    filtered_iba1_cells : ndarray
        Labeled image of filtered IBA1+ cells
    morphology_classes : dict
        Dictionary mapping cell labels to morphology classifications
    img_shape : tuple
        Shape of the original image (height, width)
        
    Returns:
    --------
    ndarray
        RGB visualization of cell morphology
    """
    # Create empty RGB image at full resolution
    height, width = img_shape
    print(f"Creating morphology visualization of size {width}x{height}...")
    
    # Convert to numpy array for faster operations
    filtered_iba1_np = np.asarray(filtered_iba1_cells)
    
    # Create a mapping from label to class as a numpy array for faster lookup
    max_label = int(np.max(filtered_iba1_np))
    label_class_map = np.zeros(max_label + 1, dtype=np.uint8)
    
    # Map class names to integers for faster processing
    class_to_int = {
        'ameboid': 1,
        'rod': 2,
        'ramified': 3,
        'hyperramified': 4,
        'artifact': 5,
        'unclassified': 6
    }
    
    # Populate the label->class map
    print("Building label->class mapping...")
    for label, info in morphology_classes.items():
        class_name = info['classification']
        label_class_map[int(label)] = class_to_int.get(class_name, 6)  # Default to unclassified
    
    # Pre-allocate the visualization array
    morph_vis = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Define colors for each class (RGB format, pre-multiplied to half intensity)
    class_colors = {
        1: [128, 38, 38],    # ameboid (Red)
        2: [128, 77, 0],     # rod (Orange)
        3: [38, 128, 38],    # ramified (Green)
        4: [38, 38, 128],    # hyperramified (Blue)
        5: [128, 38, 128],   # artifact (Purple)
        6: [89, 89, 89]      # unclassified (Gray)
    }
    
    # Process each morphology class separately (much faster than processing by label)
    print("Building morphology visualization by class...")
    
    # First, create temporary arrays for each class
    for class_id in range(1, 7):  # 1-6 for the different classes
        # Create a mask for all cells of this class
        class_labels = [int(label) for label, info in morphology_classes.items() 
                       if class_to_int.get(info['classification'], 6) == class_id]
        
        if class_labels:
            print(f"Processing {len(class_labels)} cells for class {class_id}...")
            
            # Process in batches to avoid memory issues
            batch_size = 1000
            class_mask = np.zeros_like(filtered_iba1_np, dtype=bool)
            
            for i in range(0, len(class_labels), batch_size):
                batch_labels = class_labels[i:i+batch_size]
                
                # Create batch mask with optimized numpy operations
                batch_mask = np.isin(filtered_iba1_np, batch_labels)
                class_mask |= batch_mask
            
            # Apply color to all cells in this class at once
            color = class_colors[class_id]
            for c in range(3):
                morph_vis[:,:,c][class_mask] = color[c]
    
    # Calculate boundaries more efficiently
    print("Calculating cell boundaries...")
    
    # Function to find boundaries more efficiently
    def fast_find_boundaries(label_image):
        """Faster boundary finding than segmentation.find_boundaries"""
        # Get image with shifted labels (right and down)
        shifted_right = np.zeros_like(label_image)
        shifted_right[:, :-1] = label_image[:, 1:]
        
        shifted_down = np.zeros_like(label_image)
        shifted_down[:-1, :] = label_image[1:, :]
        
        # Find boundary pixels (where labels change)
        boundaries = (label_image != shifted_right) | (label_image != shifted_down)
        
        # Don't mark background-to-background transitions as boundaries
        boundaries &= (label_image > 0) | (shifted_right > 0) | (shifted_down > 0)
        
        return boundaries
    
    # Find boundaries and apply white color
    boundaries = fast_find_boundaries(filtered_iba1_np)
    morph_vis[boundaries] = 255  # Set all channels to white
    
    # Create a legend image
    legend_fig = plt.figure(figsize=(3, 1))
    ax = legend_fig.add_subplot(111)
    
    # Create colored patches for legend
    import matplotlib.patches as mpatches
    patches = []
    
    # Map integer class IDs back to names
    int_to_class = {v: k for k, v in class_to_int.items()}
    
    # Get unique classes used
    used_classes = set()
    for label, info in morphology_classes.items():
        used_classes.add(info['classification'])
    
    # Add patches for each used class
    for class_id, color_rgb in class_colors.items():
        class_name = int_to_class.get(class_id)
        if class_name in used_classes:
            # Don't show artifacts in the legend unless explicitly requested
            if class_name == 'artifact':
                continue
                
            # Convert to 0-1 range for matplotlib
            norm_color = [c/255 for c in color_rgb]
            patches.append(mpatches.Patch(color=norm_color, label=class_name.capitalize()))
    
    # Add legend with patches
    ax.legend(handles=patches, loc='center', frameon=False)
    ax.axis('off')
    
    print("Morphology visualization completed.")
    
    return morph_vis, legend_fig
def save_results_with_qki(results, visualizations, output_dir):
    """
    Save analysis results and visualizations to disk with optimized performance
    
    Parameters:
    -----------
    results : dict
        Results dictionary from detect_nuclei_inside_iba1_with_qki_tiled
    visualizations : dict
        Visualization images from create_visualizations
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    None
    """
    import os
    import cv2
    import numpy as np
    import pandas as pd
    import time
    import concurrent.futures
    from PIL import Image
    
    start_time = time.time()
    print(f"Saving analysis results to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Function to save a single visualization
    def save_visualization(viz_tuple):
        name, img = viz_tuple
        
        try:
            # Prepare image for saving with OpenCV
            if img.dtype == np.float32 or img.dtype == np.float64:
                # Scale floating point images to 0-255
                img_save = (img * 255).astype(np.uint8)
            else:
                img_save = img.copy()
                
            # Convert to BGR if it's RGB
            if len(img_save.shape) == 3 and img_save.shape[2] == 3:
                img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
            
            # For morphology, handle specially
            if name == 'morphology':
                # Save the morphology image directly - FULL RESOLUTION
                cv2.imwrite(os.path.join(output_dir, f"{name}.png"), img_save)
                
                # Create and save a separate legend image if needed
                if results.get('morphology_legend'):
                    # Create a small legend-only image
                    legend_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
                    
                    # Draw legend text and color boxes with OpenCV
                    class_colors = {
                        'Ameboid': (50, 50, 200),      # BGR Red
                        'Ramified': (50, 200, 50),     # BGR Green
                        'Hyperramified': (200, 50, 50), # BGR Blue
                        'Rod': (0, 100, 200)           # BGR Orange
                    }
                    
                    y_pos = 50
                    for class_name, color in class_colors.items():
                        # Draw color box
                        cv2.rectangle(legend_img, (50, y_pos), (100, y_pos+30), color, -1)
                        
                        # Add text
                        cv2.putText(legend_img, class_name, (120, y_pos+20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        
                        y_pos += 60
                    
                    # Save the legend
                    cv2.imwrite(os.path.join(output_dir, f"{name}_legend.png"), legend_img)
            else:
                # Save all other images directly with OpenCV
                cv2.imwrite(os.path.join(output_dir, f"{name}.png"), img_save)
                
            return True
        except Exception as e:
            print(f"Error saving {name} image: {e}")
            # Fallback for unusual image types
            try:
                # Try with PIL as fallback
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # Convert to RGB for PIL
                    pil_img = Image.fromarray(img_save if img_save.dtype == np.uint8 else img_save.astype(np.uint8))
                    pil_img.save(os.path.join(output_dir, f"{name}.png"))
                else:
                    # Grayscale
                    pil_img = Image.fromarray(img_save if img_save.dtype == np.uint8 else img_save.astype(np.uint8), mode='L')
                    pil_img.save(os.path.join(output_dir, f"{name}.png"))
                return True
            except Exception as e2:
                print(f"Failed to save {name} with fallback method: {e2}")
                return False
    
    # Parallel save visualizations using ThreadPoolExecutor
    print("Saving visualizations in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all visualization saves
        future_to_viz = {executor.submit(save_visualization, (name, img)): name 
                         for name, img in visualizations.items()}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_viz):
            name = future_to_viz[future]
            try:
                saved = future.result()
                if saved:
                    print(f"  - Saved {name} visualization")
            except Exception as exc:
                print(f"  - Error saving {name}: {exc}")
    
    # Save CSV data in parallel
    print("Saving CSV data in parallel...")
    
    # Prepare all CSV data for saving
    csv_tasks = []
    
    # Stats CSV
    if 'stats' in results:
        stats_df = pd.DataFrame([results['stats']])
        csv_tasks.append((stats_df, os.path.join(output_dir, "stats.csv")))
    
    # Nuclei properties CSV
    if 'nuclei_properties' in results:
        nuclei_df = pd.DataFrame(results['nuclei_properties'])
        csv_tasks.append((nuclei_df, os.path.join(output_dir, "nuclei_properties.csv")))
    
    # Morphology classification CSV if available
    if 'stats' in results and 'morphology_counts' in results['stats']:
        morph_counts = results['stats']['morphology_counts']
        morph_df = pd.DataFrame({
            'morphology_class': list(morph_counts.keys()),
            'count': list(morph_counts.values()),
            'percentage': [count / sum(morph_counts.values()) * 100 for count in morph_counts.values()]
        })
        csv_tasks.append((morph_df, os.path.join(output_dir, "morphology_classification.csv")))
    
    # Function to save a single CSV file
    def save_csv(task):
        df, path = task
        try:
            df.to_csv(path, index=False)
            return os.path.basename(path)
        except Exception as e:
            print(f"Error saving CSV {path}: {e}")
            return None
    
    # Save CSVs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all CSV saves
        future_to_csv = {executor.submit(save_csv, task): task[1] for task in csv_tasks}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_csv):
            path = future_to_csv[future]
            try:
                result = future.result()
                if result:
                    print(f"  - Saved {result}")
            except Exception as exc:
                print(f"  - Error saving {os.path.basename(path)}: {exc}")
    
    # Create a pie chart using OpenCV for morphology classes
    if 'stats' in results and 'morphology_counts' in results['stats']:
        print("Creating morphology pie chart...")
        
        # Get pie chart data
        morph_counts = results['stats']['morphology_counts']
        counts = np.array(list(morph_counts.values()))
        labels = list(morph_counts.keys())
        
        if sum(counts) > 0:  # Only if we have data
            percentages = counts / sum(counts) * 100
            
            # Create a blank image - use high resolution
            width, height = 1200, 1200
            pie_img = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Colors for each class (in BGR)
            colors = [
                (85, 85, 255),   # Ameboid - BGR for Red
                (0, 102, 255),   # Rod - BGR for Orange
                (85, 255, 85),   # Ramified - BGR for Green
                (255, 85, 85)    # Hyperramified - BGR for Blue
            ]
            
            # Draw pie chart
            center = (width // 2, height // 2)
            radius = min(width, height) // 3
            
            # Calculate angles for each section
            angles = []
            start_angle = 0
            for percentage in percentages:
                angle = 360 * percentage / 100
                angles.append((start_angle, start_angle + angle))
                start_angle += angle
            
            # Draw each section of the pie
            for i, (start, end) in enumerate(angles):
                # Draw sector
                cv2.ellipse(pie_img, center, (radius, radius), 0, start, end, colors[i % len(colors)], -1)
                
                # Calculate text position (middle of sector)
                middle_angle = (start + end) / 2 * np.pi / 180
                text_x = int(center[0] + (radius * 1.3) * np.cos(middle_angle))
                text_y = int(center[1] + (radius * 1.3) * np.sin(middle_angle))
                
                # Add label and percentage
                label_text = f"{labels[i]}: {percentages[i]:.1f}%"
                
                # Adjust text position based on angle
                if 90 < middle_angle * 180 / np.pi < 270:
                    text_align = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(pie_img, label_text, (text_x - 150, text_y), text_align, 1, (0, 0, 0), 2)
                else:
                    text_align = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(pie_img, label_text, (text_x, text_y), text_align, 1, (0, 0, 0), 2)
            
            # Add title
            cv2.putText(pie_img, "IBA1+ Cell Morphology Distribution", (width // 2 - 300, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            
            # Save the pie chart
            cv2.imwrite(os.path.join(output_dir, "morphology_pie_chart.png"), pie_img)
            print("  - Saved morphology pie chart")
    
    # If available in the results, also save the correlation plots
    if 'morphology_classes' in results and results.get('nuclei_properties'):
        print("Creating and saving correlation plots...")
        save_correlation_plots(results, output_dir)
    
    # Generate unified comprehensive cell report
    unified_df = save_unified_cell_report(results, output_dir)
    
    total_time = time.time() - start_time
    print(f"Results saved to {output_dir} in {total_time:.2f} seconds")


def save_correlation_plots(results, output_dir):
    """
    Creates and saves correlation plots between morphology and Qki expression
    using OpenCV to maintain full resolution.
    
    Parameters:
    -----------
    results : dict
        Results dictionary containing cell data
    output_dir : str
        Directory to save the plots
        
    Returns:
    --------
    None
    """
    # Skip if necessary data isn't available
    if 'morphology_classes' not in results or not results.get('nuclei_properties'):
        return

    # Extract cell data
    cell_df = None
    
    # Try different ways to get the cell dataframe
    if 'unified_cell_report' in results:
        cell_df = results['unified_cell_report']
    else:
        # Try to recreate cell data from available information
        cell_data = []
        for prop in results['iba1_props']:
            cell_label = prop.label
            
            # Skip cells not in morphology classification
            if cell_label not in results['morphology_classes']:
                continue
                
            # Get morphology data
            morph_data = results['morphology_classes'][cell_label]
            
            # Simple cell location
            y, x = prop.centroid
            
            # Add cell data
            cell_data.append({
                'cell_id': cell_label,
                'cell_type': 'iba1',
                'morphology_class': morph_data['classification'],
                'area': prop.area,
                'perimeter': prop.perimeter,
                'solidity': prop.solidity,
                'eccentricity': prop.eccentricity,
                'centroid_y': y,
                'centroid_x': x
            })
        
        if cell_data:
            cell_df = pd.DataFrame(cell_data)
        else:
            print("No cell data available for correlation plots")
            return
    
    # Group data by morphology
    morph_nuclear_qki = {'ameboid': [], 'ramified': [], 'hyperramified': []}
    morph_cyto_qki = {'ameboid': [], 'ramified': [], 'hyperramified': []}
    morph_nc_ratio = {'ameboid': [], 'ramified': [], 'hyperramified': []}
    
    # Extract Qki metrics by morphology for IBA1 cells only
    iba1_cells = cell_df[cell_df['cell_type'] == 'iba1']
    for morph in ['ameboid', 'ramified', 'hyperramified']:
        if 'avg_nuclear_qki' in iba1_cells.columns:
            morph_cells = iba1_cells[iba1_cells['morphology_class'] == morph]
            if not morph_cells.empty:
                morph_nuclear_qki[morph] = morph_cells['avg_nuclear_qki'].dropna().tolist()
                if 'avg_cytoplasmic_qki' in morph_cells.columns:
                    morph_cyto_qki[morph] = morph_cells['avg_cytoplasmic_qki'].dropna().tolist()
                if 'avg_nc_ratio' in morph_cells.columns:
                    morph_nc_ratio[morph] = morph_cells['avg_nc_ratio'].dropna().tolist()
    
    # Create a custom OpenCV-based visualization for the comparison
    # Set up a high-resolution image
    width, height = 1800, 600
    corr_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Define the three plot regions
    plot_width = width // 3
    plot_height = height - 100  # Leave space for titles
    
    # Draw the plots using OpenCV
    plot_titles = ["Nuclear Qki by Cell Morphology", 
                  "Cytoplasmic Qki by Cell Morphology", 
                  "N:C Ratio by Cell Morphology"]
    
    plot_data = [
        morph_nuclear_qki,
        morph_cyto_qki,
        morph_nc_ratio
    ]
    
    for plot_idx in range(3):
        # Calculate the x-offset for this plot
        x_offset = plot_idx * plot_width
        
        # Draw plot title
        cv2.putText(corr_img, plot_titles[plot_idx], 
                   (x_offset + 20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Draw Y-axis line
        cv2.line(corr_img, 
                (x_offset + 100, 100), 
                (x_offset + 100, height - 100), 
                (0, 0, 0), 2)
        
        # Draw X-axis line
        cv2.line(corr_img, 
                (x_offset + 100, height - 100), 
                (x_offset + plot_width - 50, height - 100), 
                (0, 0, 0), 2)
        
        # Label x-axis (categories)
        categories = ['Ameboid', 'Ramified', 'Hyperramified']
        for i, cat in enumerate(categories):
            x_pos = x_offset + 150 + i * 150
            cv2.putText(corr_img, cat, 
                       (x_pos - 40, height - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Draw tick marks
            cv2.line(corr_img, 
                    (x_pos, height - 100), 
                    (x_pos, height - 95), 
                    (0, 0, 0), 2)
        
        # Get data for this plot
        data_dict = plot_data[plot_idx]
        
        # Find min/max for scaling Y-axis
        all_values = []
        for values in data_dict.values():
            all_values.extend(values)
        
        if not all_values:
            continue  # Skip if no data
            
        data_min = min(all_values) if all_values else 0
        data_max = max(all_values) if all_values else 1
        
        # Add padding to min/max
        data_range = data_max - data_min
        data_min -= data_range * 0.1
        data_max += data_range * 0.1
        
        # Draw Y-axis labels
        for i in range(5):
            value = data_min + (data_max - data_min) * i / 4
            y_pos = int(height - 100 - (i / 4) * (height - 200))
            
            # Draw value label
            cv2.putText(corr_img, f"{value:.1f}", 
                       (x_offset + 50, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Draw tick mark
            cv2.line(corr_img, 
                    (x_offset + 95, y_pos), 
                    (x_offset + 100, y_pos), 
                    (0, 0, 0), 1)
        
        # Draw boxplots
        colors = [(0, 0, 200), (0, 200, 0), (200, 0, 0)]  # BGR colors
        
        for i, (cat, values) in enumerate(data_dict.items()):
            if not values:
                continue
                
            # Calculate box plot statistics
            q1 = np.percentile(values, 25)
            q2 = np.percentile(values, 50)  # median
            q3 = np.percentile(values, 75)
            
            # Calculate whiskers
            iqr = q3 - q1
            lower_whisker = max(min(values), q1 - 1.5 * iqr)
            upper_whisker = min(max(values), q3 + 1.5 * iqr)
            
            # Scale to pixel positions
            y_scale = (height - 200) / (data_max - data_min)
            
            # Calculate pixel positions
            x_center = x_offset + 150 + i * 150
            y_q1 = int(height - 100 - (q1 - data_min) * y_scale)
            y_q2 = int(height - 100 - (q2 - data_min) * y_scale)
            y_q3 = int(height - 100 - (q3 - data_min) * y_scale)
            y_lower = int(height - 100 - (lower_whisker - data_min) * y_scale)
            y_upper = int(height - 100 - (upper_whisker - data_min) * y_scale)
            
            # Draw box (rectangle)
            cv2.rectangle(corr_img, 
                         (x_center - 25, y_q3), 
                         (x_center + 25, y_q1), 
                         colors[i], 2)
            
            # Draw median line
            cv2.line(corr_img, 
                    (x_center - 25, y_q2), 
                    (x_center + 25, y_q2), 
                    (0, 0, 0), 2)
            
            # Draw whiskers
            # Upper whisker
            cv2.line(corr_img, 
                    (x_center, y_q3), 
                    (x_center, y_upper), 
                    colors[i], 1)
            # Lower whisker
            cv2.line(corr_img, 
                    (x_center, y_q1), 
                    (x_center, y_lower), 
                    colors[i], 1)
            
            # Draw caps on whiskers
            cv2.line(corr_img, 
                    (x_center - 10, y_upper), 
                    (x_center + 10, y_upper), 
                    colors[i], 1)
            cv2.line(corr_img, 
                    (x_center - 10, y_lower), 
                    (x_center + 10, y_lower), 
                    colors[i], 1)
            
            # Draw outliers if any
            for value in values:
                if value < lower_whisker or value > upper_whisker:
                    y_outlier = int(height - 100 - (value - data_min) * y_scale)
                    cv2.circle(corr_img, 
                              (x_center, y_outlier), 
                              3, colors[i], -1)
    
    # Save the morphology correlation image
    cv2.imwrite(os.path.join(output_dir, "morphology_qki_correlation.png"), corr_img)
    print("  - Saved morphology-QKI correlation plots")


def save_detailed_report(results, output_dir):
    """
    Saves a detailed report of cell analysis results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from detect_nuclei_inside_iba1_with_qki_tiled
    output_dir : str
        Directory to save the report
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with detailed results
    """
    import os
    import pandas as pd
    import numpy as np
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating detailed report...")
    
    # Create a comprehensive report
    report_data = {
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'processing_time_seconds': results['stats']['processing_time'],
        'image_dimensions': f"{results['image'].shape[1]}x{results['image'].shape[0]}",
        'number_of_channels': results['image'].shape[2],
        'iba1_cells_count': results['stats']['iba1_count'],
        'dapi_nuclei_count': results['stats']['nuclei_count'],
        'nuclei_in_iba1_count': results['stats']['nuclei_in_iba1_count'],
        'avg_nuclei_per_iba1': results['stats']['avg_nuclei_per_iba1'],
        'avg_nuclear_qki': results['stats']['avg_nuclear_qki'],
        'avg_cytoplasmic_qki': results['stats']['avg_cyto_qki'],
        'avg_nc_ratio': results['stats']['avg_nc_ratio']
    }
    
    # Add morphology counts
    if 'morphology_counts' in results['stats']:
        for morph_class, count in results['stats']['morphology_counts'].items():
            if morph_class != 'artifact':  # Skip artifacts in main counts
                report_data[f'{morph_class}_cells_count'] = count
                
                # Calculate percentage
                percentage = count / results['stats']['iba1_count'] * 100 if results['stats']['iba1_count'] > 0 else 0
                report_data[f'{morph_class}_cells_percentage'] = percentage
        
        # Add artifact count separately
        if 'artifact' in results['stats']['morphology_counts']:
            report_data['potential_artifacts_count'] = results['stats']['morphology_counts']['artifact']
    
    # Create dataframe
    report_df = pd.DataFrame([report_data])
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "detailed_analysis_report.csv")
    report_df.to_csv(csv_path, index=False)
    print(f"Detailed report saved to: {csv_path}")
    
    # Create statistical summaries for various metrics
    
    # IBA1 cell metrics
    if results.get('iba1_props'):
        iba1_metrics = {
            'metric': ['area', 'perimeter', 'solidity', 'eccentricity'],
            'min': [],
            'max': [],
            'mean': [],
            'median': [],
            'std': []
        }
        
        for metric in iba1_metrics['metric']:
            values = [getattr(prop, metric) for prop in results['iba1_props'] if hasattr(prop, metric)]
            if values:
                iba1_metrics['min'].append(np.min(values))
                iba1_metrics['max'].append(np.max(values))
                iba1_metrics['mean'].append(np.mean(values))
                iba1_metrics['median'].append(np.median(values))
                iba1_metrics['std'].append(np.std(values))
            else:
                iba1_metrics['min'].append(0)
                iba1_metrics['max'].append(0)
                iba1_metrics['mean'].append(0)
                iba1_metrics['median'].append(0)
                iba1_metrics['std'].append(0)
        
        iba1_metrics_df = pd.DataFrame(iba1_metrics)
        iba1_metrics_df.to_csv(os.path.join(output_dir, "iba1_cell_metrics_summary.csv"), index=False)
    
    # Nuclei metrics
    if results.get('nuclei_properties'):
        nuclei_metrics = {
            'metric': ['area', 'nuclear_qki', 'cytoplasmic_qki', 'nc_ratio'],
            'min': [],
            'max': [],
            'mean': [],
            'median': [],
            'std': []
        }
        
        for metric in nuclei_metrics['metric']:
            values = [nucleus[metric] for nucleus in results['nuclei_properties'] 
                     if metric in nucleus and nucleus[metric] != float('inf')]
            if values:
                nuclei_metrics['min'].append(np.min(values))
                nuclei_metrics['max'].append(np.max(values))
                nuclei_metrics['mean'].append(np.mean(values))
                nuclei_metrics['median'].append(np.median(values))
                nuclei_metrics['std'].append(np.std(values))
            else:
                nuclei_metrics['min'].append(0)
                nuclei_metrics['max'].append(0)
                nuclei_metrics['mean'].append(0)
                nuclei_metrics['median'].append(0)
                nuclei_metrics['std'].append(0)
        
        nuclei_metrics_df = pd.DataFrame(nuclei_metrics)
        nuclei_metrics_df.to_csv(os.path.join(output_dir, "nuclei_metrics_summary.csv"), index=False)
    
    # Include experimental parameters
    params = {
        'parameter': [
            'iba1_min_size', 'iba1_max_size', 
            'dapi_min_size', 'dapi_max_size',
            'artifact_area_threshold', 
            'tile_size', 'overlap'
        ],
        'value': [
            results.get('artifact_filtering', {}).get('iba1_min_size', 'N/A'),
            results.get('artifact_filtering', {}).get('iba1_max_size', 'N/A'),
            results.get('artifact_filtering', {}).get('dapi_min_size', 'N/A'),
            results.get('artifact_filtering', {}).get('dapi_max_size', 'N/A'),
            results.get('artifact_filtering', {}).get('area_threshold', 'N/A'),
            'N/A',  # tile_size not stored in results
            'N/A'   # overlap not stored in results
        ]
    }
    
    params_df = pd.DataFrame(params)
    params_df.to_csv(os.path.join(output_dir, "analysis_parameters_summary.csv"), index=False)
    
    return report_df
def main(image_path, output_dir, 
         iba1_channel=1, dapi_channel=2, qki_channel=0,
         iba1_min_size=50, iba1_max_size=5000,
         dapi_min_size=15, dapi_max_size=500,
         iba1_threshold_method="adaptive", dapi_threshold_method="otsu",
         use_enhanced_iba1=True,
         remove_artifacts=True, artifact_area_threshold=8000,
         iba1_min_intensity=None, iba1_max_intensity=None,
         tile_size=2000, overlap=200, max_workers=4):
    """
    Main function to run the entire analysis pipeline for IBA1+, DAPI+ nuclei and Qki expression
    
    Parameters:
    -----------
    image_path : str
        Path to the multi-channel image file
    output_dir : str
        Directory to save the results
    iba1_channel, dapi_channel, qki_channel : int
        Channel indices for staining (0-indexed)
    iba1_min_size, iba1_max_size, dapi_min_size, dapi_max_size : int
        Size filters for cells and nuclei in pixels
    iba1_threshold_method, dapi_threshold_method : str
        Methods for thresholding
    use_enhanced_iba1 : bool
        Whether to use the enhanced detection algorithm
    remove_artifacts : bool
        Whether to apply additional artifact removal steps
    artifact_area_threshold : int
        Size threshold above which objects are considered potential artifacts
    iba1_min_intensity, iba1_max_intensity : float or None
        Intensity thresholds for cells
    tile_size, overlap : int
        Tiling parameters
    max_workers : int
        Number of parallel workers to process tiles
        
    Returns:
    --------
    dict
        Dictionary containing all processing results and measurements
    """
    start_time = time.time()
    print(f"Starting analysis pipeline for {image_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Log analysis parameters
    with open(os.path.join(output_dir, "analysis_parameters.txt"), 'w') as f:
        f.write(f"Analysis parameters for {os.path.basename(image_path)}:\n")
        f.write(f"Channels: IBA1={iba1_channel}, DAPI={dapi_channel}, QKI={qki_channel}\n")
        f.write(f"Size filters: IBA1={iba1_min_size}-{iba1_max_size}, DAPI={dapi_min_size}-{dapi_max_size}\n")
        f.write(f"Threshold methods: IBA1={iba1_threshold_method}, DAPI={dapi_threshold_method}\n")
        f.write(f"Enhanced detection: IBA1={use_enhanced_iba1}\n")
        f.write(f"Artifact removal: {remove_artifacts}, threshold={artifact_area_threshold}\n")
        f.write(f"Intensity filters: IBA1={iba1_min_intensity}-{iba1_max_intensity}\n")
        f.write(f"Tiling parameters: size={tile_size}, overlap={overlap}, workers={max_workers}\n")
        f.write(f"Analysis started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Run cell detection and analysis
    results = detect_nuclei_inside_iba1_with_qki_tiled(
        image_path=image_path,
        iba1_channel=iba1_channel,
        dapi_channel=dapi_channel,
        qki_channel=qki_channel,
        iba1_threshold_method=iba1_threshold_method,
        dapi_threshold_method=dapi_threshold_method,
        iba1_min_size=iba1_min_size,
        iba1_max_size=iba1_max_size,
        dapi_min_size=dapi_min_size,
        dapi_max_size=dapi_max_size,
        boundary_thickness=2,
        cytoplasm_expansion=3,
        tile_size=tile_size,
        overlap=overlap,
        max_workers=max_workers,
        use_enhanced_iba1=use_enhanced_iba1,
        remove_artifacts=remove_artifacts,
        artifact_area_threshold=artifact_area_threshold,
        artifact_solidity_threshold=0.95,
        iba1_min_intensity=iba1_min_intensity,
        iba1_max_intensity=iba1_max_intensity
    )
    
    # Create visualizations
    print("Creating visualizations...")
    visualizations = create_visualizations(results)
    
    # Save results
    print("Saving results...")
    save_results_with_qki(results, visualizations, output_dir)
    
    # Generate detailed report
    print("Generating detailed report...")
    detailed_df = save_detailed_report(results, output_dir)
    
    # Generate unified comprehensive cell report with all metrics in one CSV
    print("Generating unified cell report...")
    unified_df = save_unified_cell_report(results, output_dir)
    
    # Print final summary
    total_time = time.time() - start_time
    print("\nFinal Summary:")
    print(f"IBA1+ cells: {results['stats']['iba1_count']}")
    print(f"DAPI+ nuclei: {results['stats']['nuclei_count']}")
    print(f"Analysis completed in {total_time/60:.1f} minutes")
    
    return results


if __name__ == "__main__":
    # Example usage
    image_path = "/path/to/your/3channel_composite_image.tif"
    output_dir = "/path/to/output/directory"
    
    # Run the analysis with 3 channels: DAPI (0), IBA1 (1), QKI (2)
    results = main(
        image_path=image_path,
        output_dir=output_dir,
        dapi_channel=0,
        iba1_channel=1,
        qki_channel=2,
        iba1_threshold_method="adaptive",
        iba1_min_size=50,
        iba1_max_size=5000,
        dapi_min_size=15,
        dapi_max_size=500,
        tile_size=2000,
        overlap=200,
        max_workers=16,
        use_enhanced_iba1=True,
        remove_artifacts=True,
        artifact_area_threshold=8000,
        iba1_min_intensity=None,
        iba1_max_intensity=None
    )
    
    print("Analysis complete!")