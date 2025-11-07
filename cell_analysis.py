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
            'cell_type': 'iba1',  # Add cell type to differentiate from T cells
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
    
    # NEW: Add T cells if available in results
    if 'tcell_props' in results and results['tcell_props']:
        tcell_props = results['tcell_props']
        tcell_labels = results['tcell_labels']
        
        for prop in tcell_props:
            cell_label = prop.label
            
            # Basic cell properties
            y, x = prop.centroid
            area = prop.area
            perimeter = prop.perimeter
            solidity = prop.solidity
            eccentricity = prop.eccentricity
            
            # Create T cell data dictionary
            tcell_data = {
                'cell_id': cell_label,
                'cell_type': 'tcell',  # Identify as T cell
                'centroid_x': x,
                'centroid_y': y,
                'area': area,
                'perimeter': perimeter,
                'solidity': solidity,
                'eccentricity': eccentricity,
                'morphology_class': 'tcell',  # No morphology classification for T cells
                'has_nuclei': False,  # Not calculating nuclei for T cells
                'nuclei_count': 0,
                'avg_nuclear_qki': 0,
                'avg_cytoplasmic_qki': 0,
                'avg_nc_ratio': 0,
            }
            
            # Add cell to the list
            all_cells_data.append(tcell_data)
    
    # Create DataFrame and save to CSV
    if all_cells_data:
        df = pd.DataFrame(all_cells_data)
        csv_path = os.path.join(output_dir, "all_cells_comprehensive.csv")
        df.to_csv(csv_path, index=False)
        print(f"Unified cell report saved to: {csv_path}")
        
        # Also create separate reports for cell types
        iba1_cells = df[df['cell_type'] == 'iba1']
        tcells = df[df['cell_type'] == 'tcell']
        
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
        
        # Save separate CSV for T cells
        if not tcells.empty:
            tcells.to_csv(os.path.join(output_dir, "tcells.csv"), index=False)
            
        # NEW: Save cell centers for spatial analysis
        centers_df = df[['cell_type', 'centroid_x', 'centroid_y']]
        centers_df.to_csv(os.path.join(output_dir, "cell_centers.csv"), index=False)
        
        # Save separate center files for each cell type
        iba1_centers = centers_df[centers_df['cell_type'] == 'iba1']
        tcell_centers = centers_df[centers_df['cell_type'] == 'tcell']
        
        if not iba1_centers.empty:
            iba1_centers.to_csv(os.path.join(output_dir, "iba1_cell_centers.csv"), index=False)
        
        if not tcell_centers.empty:
            tcell_centers.to_csv(os.path.join(output_dir, "tcell_centers.csv"), index=False)
            
        return df
    else:
        print("No cells found for unified report.")
        return None

def process_tile(args):
    """
    Ultra-optimized tile processing with all required keys in the output dictionary.
    """
    (tile_img, tile_coords, iba1_channel, dapi_channel, qki_channel, tcell_channel,
     iba1_threshold_method, dapi_threshold_method, iba1_min_size, 
     iba1_max_size, dapi_min_size, dapi_max_size, tcell_min_size, 
     tcell_max_size, cytoplasm_expansion) = args
    
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
    
    tcell_mask = np.zeros_like(dapi_img, dtype=bool)
    if tcell_channel < tile_img.shape[2]:
        tcell_img = tile_img[:, :, tcell_channel]
        tcell_mask = tcell_img > np.percentile(tcell_img, 80)
    
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
    
    # Minimal T-cell processing
    tcell_props = []
    tcell_labels = None
    if np.any(tcell_mask):
        tcell_labels, _ = ndimage.label(tcell_mask)
        # Just get basic properties
        for region in measure.regionprops(tcell_labels):
            if tcell_min_size <= region.area <= tcell_max_size:
                # Create a minimal property dictionary
                prop_dict = {
                    'label': region.label,
                    'area': region.area,
                    'perimeter': 0,  # Placeholder
                    'solidity': 0,   # Placeholder
                    'eccentricity': 0,  # Placeholder
                    'centroid_y': region.centroid[0],
                    'centroid_x': region.centroid[1]
                }
                tcell_props.append(prop_dict)
    
    # Return ALL required keys
    return {
        'coords': tile_coords,
        'iba1_mask': iba1_mask,                 # Required key
        'nuclei_mask': dapi_binary,             # Required key
        'nuclei_in_iba1_mask': nuclei_in_iba1,  # Required key
        'nuclei_props': nuclei_props,           # Required key
        'tcell_mask': tcell_mask,               # Required key
        'tcell_props': tcell_props,             # Required key
        'tcell_labels': tcell_labels            # May be required
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
    
    # NEW: Additional intensity-based filtering
    valid_labels = []
    for prop in props:
        # Basic size filter
        if not (min_size <= prop.area <= max_size):
            continue
            
        # NEW: Intensity-based filtering
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


# NEW FUNCTION: Added a similar function for T cell detection
def enhance_tcell_detection(tcell_img, min_size=20, max_size=2000, 
                           remove_large_artifacts=True, artifact_threshold=5000,
                           min_intensity=None, max_intensity=None):
    """
    Advanced T cell detection with multi-scale processing and cell reconstruction.
    Similar approach to IBA1 detection but tuned for T cell characteristics.
    
    Parameters as in enhance_iba1_detection, but defaults adjusted for T cells
    which tend to be smaller and more rounded than IBA1 cells.
    
    Returns:
    --------
    ndarray
        Binary mask of detected T cells
    """
    # Normalize to 0-255 range
    tcell_normalized = cv2.normalize(tcell_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    tcell_enhanced = clahe.apply(tcell_normalized)
    
    # Gaussian blur to reduce noise
    tcell_blurred = cv2.GaussianBlur(tcell_enhanced, (5, 5), 1)
    
    # Background subtraction using rolling ball algorithm
    selem_size = max(15, int(min(tcell_blurred.shape) * 0.03))  # Smaller for T cells
    selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (selem_size, selem_size))
    
    background = cv2.morphologyEx(tcell_blurred, cv2.MORPH_OPEN, selem)
    tcell_no_bg = cv2.subtract(tcell_blurred, background)
    
    # Contrast enhancement
    tcell_stretched = cv2.normalize(tcell_no_bg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Global thresholding
    try:
        thresh_value = filters.threshold_otsu(tcell_stretched)
    except:
        # Fallback to percentile-based threshold
        thresh_value = np.percentile(tcell_stretched, 75) * 0.7
        
    global_binary = tcell_stretched > thresh_value
    
    # Use adaptive thresholding for local variations
    block_size = max(21, int(min(tcell_img.shape) * 0.02))  # Smaller block size for T cells
    if block_size % 2 == 0:
        block_size += 1
        
    try:
        adaptive_binary = cv2.adaptiveThreshold(
            tcell_stretched,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            -2  # Less aggressive than IBA1
        )
        
        # Combine global and adaptive thresholding
        adaptive_binary = adaptive_binary.astype(np.uint8)
        global_binary_uint8 = global_binary.astype(np.uint8) * 255
        combined_binary = cv2.bitwise_or(global_binary_uint8, adaptive_binary)
        binary_mask = combined_binary.astype(bool)
    except Exception as e:
        print(f"Adaptive thresholding failed for T cells: {e}")
        binary_mask = global_binary
    
    # Morphological operations for T cells
    # Fill holes first (T cells often have clear centers)
    binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=50)
    
    # Remove small objects
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=20)
    
    # Watershed segmentation for better cell separation
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # Find peaks for watershed markers
    local_max_coords = feature.peak_local_max(
        distance, 
        min_distance=10,  # Smaller minimum distance for T cells
        exclude_border=False,
        labels=binary_mask
    )
    
    # Create marker mask
    local_max = np.zeros_like(distance, dtype=bool)
    for coord in local_max_coords:
        local_max[coord[0], coord[1]] = True
    
    markers = measure.label(local_max)
    
    # Apply watershed
    watershed_labels = segmentation.watershed(-distance, markers, mask=binary_mask)
    
    # Filter by size and intensity
    props = measure.regionprops(watershed_labels, intensity_image=tcell_img)
    
    valid_labels = []
    for prop in props:
        # Basic size filter
        if not (min_size <= prop.area <= max_size):
            continue
            
        # Intensity-based filtering
        mean_intensity = prop.mean_intensity
        
        if min_intensity is not None and mean_intensity < min_intensity:
            continue
            
        if max_intensity is not None and mean_intensity > max_intensity:
            continue
            
        # Shape-based filtering for artifacts
        if remove_large_artifacts and prop.area > artifact_threshold:
            solidity = prop.solidity
            circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
            
            # T cells are generally more circular and solid than IBA1 cells
            if solidity < 0.3:  # T cells usually have higher solidity
                continue
            
            # Very elongated shapes are unlikely to be T cells
            if prop.major_axis_length > 0 and prop.minor_axis_length / prop.major_axis_length < 0.2:
                continue
        
        valid_labels.append(prop.label)
    
    # Create final mask
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


# NEW FUNCTION: Similar wrapper for T cell detection
def enhance_tcell_detection_in_tile(tile_img, tcell_channel, min_size=20, max_size=2000,
                             remove_large_artifacts=True, artifact_threshold=5000,
                             min_intensity=None, max_intensity=None):
    """
    Wrapper function to apply enhanced T cell detection to a tile with artifact removal
    and intensity-based filtering
    
    Parameters:
    -----------
    tile_img : ndarray
        Input tile image with multiple channels
    tcell_channel : int
        Index of T cell channel
    min_size, max_size : int
        Size filters for T cells
    remove_large_artifacts : bool
        Whether to apply additional filtering for large artifacts
    artifact_threshold : int
        Size threshold above which objects are considered potential artifacts
    min_intensity : float or None
        Minimum mean intensity threshold for cells (in raw image values)
    max_intensity : float or None
        Maximum mean intensity threshold for cells (in raw image values)
        
    Returns:
    --------
    ndarray
        Binary mask of detected T cells
    """
    # Check if there are enough channels
    if tcell_channel >= tile_img.shape[2]:
        print(f"Warning: T cell channel {tcell_channel} is out of bounds for image with {tile_img.shape[2]} channels")
        return np.zeros((tile_img.shape[0], tile_img.shape[1]), dtype=bool)
    
    # Extract T cell channel
    tcell_img = tile_img[:, :, tcell_channel]
    
    # Apply enhanced detection
    return enhance_tcell_detection(
        tcell_img,
        min_size,
        max_size,
        remove_large_artifacts,
        artifact_threshold,
        min_intensity,
        max_intensity
    )


def detect_nuclei_inside_iba1_with_qki_tiled(image_path, iba1_channel=1, dapi_channel=2, qki_channel=0,
                                     tcell_channel=3,  # NEW: Added T cell channel parameter
                                     iba1_threshold_method="adaptive", dapi_threshold_method="otsu",
                                     iba1_min_size=50, iba1_max_size=5000,
                                     dapi_min_size=15, dapi_max_size=500,
                                     tcell_min_size=20, tcell_max_size=2000,  # NEW: T cell size parameters
                                     boundary_thickness=2, cytoplasm_expansion=3,
                                     tile_size=2000, overlap=50, max_workers=16,
                                     use_enhanced_iba1=True, use_enhanced_tcell=True,  # NEW: T cell enhancement flag
                                     remove_artifacts=True,
                                     artifact_area_threshold=8000,
                                     artifact_solidity_threshold=0.95,
                                     iba1_min_intensity=None,
                                     iba1_max_intensity=None,
                                     tcell_min_intensity=None,  # NEW: T cell intensity parameters
                                     tcell_max_intensity=None
                                    ):
    """
    Optimized tiled implementation to detect DAPI+ nuclei inside IBA1+ cells and quantify Qki expression,
    now with added T cell detection support and significantly improved performance.
    
    Parameters:
    -----------
    image_path : str
        Path to the multi-channel image file
    iba1_channel, dapi_channel, qki_channel : int
        Channel indices for staining (0-indexed)
    tcell_channel : int
        Channel index for T cell staining (0-indexed)
    iba1_threshold_method : str
        Methods for thresholding ('adaptive' is now a valid option)
    dapi_threshold_method : str
        Methods for thresholding DAPI
    iba1_min_size, iba1_max_size, dapi_min_size, dapi_max_size : int
        Size filters for cells and nuclei in pixels
    tcell_min_size, tcell_max_size : int
        Size filters for T cells in pixels
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
    use_enhanced_tcell : bool
        Whether to use the enhanced T cell detection algorithm
    remove_artifacts : bool
        Whether to apply additional artifact removal steps
    artifact_area_threshold : int
        Size threshold above which objects are considered potential artifacts
    artifact_solidity_threshold : float
        Solidity threshold for artifact identification
    iba1_min_intensity, iba1_max_intensity : float or None
        Intensity thresholds for IBA1+ cells
    tcell_min_intensity, tcell_max_intensity : float or None
        Intensity thresholds for T cells
        
    Returns:
    --------
    dict
        Dictionary containing all processing results and measurements
    """
    start_time = time.time()
    print(f"Processing image: {image_path}")
    print(f"IBA1 intensity thresholds - Min: {iba1_min_intensity}, Max: {iba1_max_intensity}")
    print(f"T cell intensity thresholds - Min: {tcell_min_intensity}, Max: {tcell_max_intensity}")
    print(f"Using 4 channels: QKI (ch{qki_channel}), IBA1 (ch{iba1_channel}), DAPI (ch{dapi_channel}), T cells (ch{tcell_channel})")
    
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image with memory-efficient method
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
    
    if max(qki_channel, iba1_channel, dapi_channel, tcell_channel) >= num_channels:
        print(f"Warning: Some channel indices are out of bounds. The image has {num_channels} channels.")
        if tcell_channel >= num_channels:
            print(f"T cell channel {tcell_channel} is not available. T cell detection will be skipped.")
    
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
    
    # OPTIMIZATION: Process enhanced T cell detection in parallel batches
    if use_enhanced_tcell and tcell_channel < num_channels:
        print("Using enhanced T cell detection algorithm with parallel processing...")
        
        # Pre-process tiles in parallel batches
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process in parallel using lambdas for simpler submission
            futures = [executor.submit(
                enhance_tcell_detection_in_tile,
                tile,
                tcell_channel,
                min_size=tcell_min_size,
                max_size=tcell_max_size,
                remove_large_artifacts=remove_artifacts,
                artifact_threshold=artifact_area_threshold // 2,  # Smaller threshold for T cells
                min_intensity=tcell_min_intensity,
                max_intensity=tcell_max_intensity
            ) for tile in tiles]
            
            # Collect results (with progress output)
            enhanced_tcell_masks = []
            for i, future in enumerate(tqdm(futures, total=len(futures), desc="Processing T cell detection")):
                enhanced_tcell_masks.append(future.result())
        
        # Create a full resolution enhanced T cell mask
        full_enhanced_tcell = np.zeros((height, width), dtype=bool)
        
        # Stitch the enhanced masks together more efficiently
        for i, mask in enumerate(enhanced_tcell_masks):
            if mask is not None:
                x_start, y_start, x_end, y_end = tile_coords[i]
                full_enhanced_tcell[y_start:y_end, x_start:x_end] |= mask
        
        # Store the enhanced mask for later use
        enhanced_results['enhanced_tcell_mask'] = full_enhanced_tcell
    
    # OPTIMIZATION: Update the process_tile arguments to include T cell parameters and
    # use simplified function calls with more optimal parameters
    process_args = [(tiles[i], tile_coords[i], iba1_channel, dapi_channel, qki_channel, tcell_channel,
                    iba1_threshold_method, dapi_threshold_method, iba1_min_size, 
                    iba1_max_size, dapi_min_size, dapi_max_size, tcell_min_size,
                    tcell_max_size, cytoplasm_expansion) 
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
    full_tcell_mask = np.zeros((height, width), dtype=bool)  # NEW: T cell mask
    
    # Collect all properties
    all_nuclei_props = []
    all_tcell_props = []  # NEW: T cell properties
    
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
        
        # NEW: Update T cell mask if available
        if result.get('tcell_mask') is not None:
            full_tcell_mask[mask_slice] |= result['tcell_mask']
        
        # Add nuclei properties
        if 'nuclei_props' in result:
            all_nuclei_props.extend(result['nuclei_props'])
        
        # NEW: Add T cell properties if available
        if 'tcell_props' in result and result['tcell_props']:
            # Adjust coordinates to global image
            for prop in result['tcell_props']:
                # Create a new property dictionary with adjusted coordinates
                prop_dict = {
                    'label': prop.label if hasattr(prop, 'label') else 0,
                    'area': prop.area if hasattr(prop, 'area') else 0,
                    'perimeter': prop.perimeter if hasattr(prop, 'perimeter') else 0,
                    'solidity': prop.solidity if hasattr(prop, 'solidity') else 0,
                    'eccentricity': prop.eccentricity if hasattr(prop, 'eccentricity') else 0,
                    'centroid_y': prop.centroid[0] + y_start if hasattr(prop, 'centroid') else 0,
                    'centroid_x': prop.centroid[1] + x_start if hasattr(prop, 'centroid') else 0
                }
                all_tcell_props.append(prop_dict)
    
    # If we have enhanced detection results, use them instead
    if 'enhanced_iba1_mask' in enhanced_results:
        print("Using enhanced IBA1 segmentation results...")
        full_iba1_mask = enhanced_results['enhanced_iba1_mask']
        
        # Recalculate nuclei in IBA1 using the enhanced mask (fast binary operation)
        full_nuclei_in_iba1_mask = full_nuclei_mask & full_iba1_mask
    
    # NEW: If we have enhanced T cell detection results, use them
    if 'enhanced_tcell_mask' in enhanced_results:
        print("Using enhanced T cell segmentation results...")
        full_tcell_mask = enhanced_results['enhanced_tcell_mask']
    
    # Additional post-processing to remove large artifacts if requested
    if remove_artifacts:
        print("Performing optimized artifact removal on full image...")
        
        # Process in chunks to reduce memory pressure (faster implementation)
        chunk_size = min(4000, height // 4)  # Adjust based on available memory
        filtered_iba1_mask = np.zeros_like(full_iba1_mask)
        filtered_tcell_mask = np.zeros_like(full_tcell_mask)
        
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
        
        # NEW: Process T cells if mask is not empty (with progress tracking) 
        if np.any(full_tcell_mask):
            for y_start in tqdm(range(0, height, chunk_size), desc="T cell artifact removal"):
                y_end = min(y_start + chunk_size, height)
                
                # Extract chunk
                chunk_mask = full_tcell_mask[y_start:y_end, :]
                
                # Skip empty chunks
                if not np.any(chunk_mask):
                    continue
                
                # Fast labeling
                chunk_labels = measure.label(chunk_mask)
                chunk_props = measure.regionprops(chunk_labels)
                
                # Filter with vectorized operations
                valid_labels = []
                for prop in chunk_props:
                    # Skip very small objects
                    if prop.area < 50:  # Lower threshold for T cells
                        valid_labels.append(prop.label)
                        continue
                        
                    # Check for artifacts with optimized logic
                    if prop.area > artifact_area_threshold / 2:  # Smaller threshold for T cells
                        # Combined metric for T cells (should be more circular)
                        circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
                        
                        # T cells should be fairly circular, not extremely large, and have reasonable solidity
                        if (prop.area > 2 * artifact_area_threshold / 2) or (prop.solidity < 0.2) or (circularity < 0.2):
                            continue
                    
                    # If not an artifact, keep it
                    valid_labels.append(prop.label)
                
                # Create valid mask with fast numpy operations
                valid_chunk_mask = np.isin(chunk_labels, valid_labels)
                
                # Update the filtered mask with this chunk
                filtered_tcell_mask[y_start:y_end, :] = valid_chunk_mask
        
        # Update the masks
        full_iba1_mask = filtered_iba1_mask
        full_tcell_mask = filtered_tcell_mask
        
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
    
    # NEW: Deduplicate T cells with the same approach
    deduplicated_tcells = []
    tcell_spatial_hash = {}
    
    if all_tcell_props:
        for tcell in all_tcell_props:
            # Skip cells with missing coordinates
            if 'centroid_x' not in tcell or 'centroid_y' not in tcell:
                continue
                
            # Create a spatial hash key
            coord_key = (int(tcell['centroid_x'] / grid_size), int(tcell['centroid_y'] / grid_size))
            
            if coord_key not in tcell_spatial_hash:
                tcell_spatial_hash[coord_key] = tcell
                deduplicated_tcells.append(tcell)
            # No need to compare metrics for T cells, just avoid duplicates
        
        print(f"Deduplicated T cell count: {len(deduplicated_tcells)} (from original {len(all_tcell_props)})")
    
    # Get the final labeled cells and nuclei (faster with skimage)
    print("Creating final labeled masks...")
    iba1_labels = measure.label(full_iba1_mask)
    nuclei_labels = measure.label(full_nuclei_mask)
    nuclei_in_iba1_labels = measure.label(full_nuclei_in_iba1_mask)
    tcell_labels = measure.label(full_tcell_mask)  # NEW: Label T cells
    
    # Analyze properties of the labeled objects (with progress tracking)
    print("Calculating region properties...")
    iba1_props = measure.regionprops(iba1_labels)
    nuclei_props = measure.regionprops(nuclei_labels)
    tcell_props = measure.regionprops(tcell_labels)  # NEW: Get T cell properties
    
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
    tcell_count = len(tcell_props)  # NEW: Count T cells
    
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
    print(f"Found {tcell_count} valid T cells")  # NEW: Report T cell count
    print(f"Qki background level: {qki_background:.1f}")
    print(f"Average nuclear Qki intensity: {avg_nuclear_qki:.2f}")
    print(f"Average cytoplasmic Qki intensity: {avg_cyto_qki:.2f}")
    print(f"Average nuclear:cytoplasmic ratio: {avg_nc_ratio:.2f}")
    
    # Assemble final results dictionary (using minimal necessary data)
    results_dict = {
        'iba1_mask': full_iba1_mask,
        'nuclei_mask': full_nuclei_mask,
        'nuclei_in_iba1_mask': full_nuclei_in_iba1_mask,
        'tcell_mask': full_tcell_mask,  # NEW: Add T cell mask
        'iba1_labels': iba1_labels,
        'nuclei_labels': nuclei_labels,
        'nuclei_in_iba1_labels': nuclei_in_iba1_labels,
        'tcell_labels': tcell_labels,  # NEW: Add T cell labels
        'nuclei_properties': deduplicated_nuclei,
        'iba1_props': iba1_props,
        'nuclei_props': nuclei_props,
        'tcell_props': tcell_props,  # NEW: Add T cell properties
        'morphology_classes': morphology_classes,
        'morphology_vis': morph_vis,
        'morphology_legend': morph_legend,
        'stats': {
            'iba1_count': iba1_count,
            'nuclei_count': nuclei_count,
            'nuclei_in_iba1_count': nuclei_in_iba1_count,
            'tcell_count': tcell_count,  # NEW: Add T cell count to stats
            'avg_nuclei_per_iba1': avg_nuclei_per_iba1,
            'qki_background': qki_background,
            'avg_nuclear_qki': avg_nuclear_qki,
            'avg_cyto_qki': avg_cyto_qki,
            'avg_nc_ratio': avg_nc_ratio,
            'processing_time': time.time() - start_time,
            'morphology_counts': morph_class_counts,
            'artifacts_removed': morph_class_counts.get('artifact', 0)  # Track artifacts
        },
        'image': img,  # Keep original image for visualizations
        'artifact_filtering': {
            'enabled': remove_artifacts,
            'area_threshold': artifact_area_threshold,
            'solidity_threshold': artifact_solidity_threshold
        },
        'intensity_filtering': {
            'iba1_min_intensity': iba1_min_intensity,
            'iba1_max_intensity': iba1_max_intensity,
            'tcell_min_intensity': tcell_min_intensity,  # NEW: Add T cell intensity info
            'tcell_max_intensity': tcell_max_intensity
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
    with corrected channel-to-color mapping and added T cell visualization
    """
    print("Creating visualizations at full resolution...")
    
    # Extract data from results
    img = results['image']
    iba1_mask = results['iba1_mask']
    nuclei_mask = results['nuclei_mask']
    nuclei_in_iba1_mask = results['nuclei_in_iba1_mask']
    iba1_labels = results['iba1_labels']
    nuclei_in_iba1_labels = results['nuclei_in_iba1_labels']
    tcell_mask = results.get('tcell_mask', None)  # NEW: Get T cell mask if available
    tcell_labels = results.get('tcell_labels', None)  # NEW: Get T cell labels if available
    morphology_vis = results.get('morphology_vis', None)
    
    # Create a composite visualization at full resolution
    composite = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    # Color mapping for 4 channels:
    # QKI (channel 0) -> Red
    # IBA1 (channel 2) -> Teal (mix of green and blue)
    # DAPI (channel 1) -> Blue
    # T cells (channel 3) -> Yellow (mix of red and green)
    
    # Extract channel indices from results
    qki_channel = 3  # Assumed to be fixed
    iba1_channel = 1
    dapi_channel = 0
    tcell_channel = 2
    
    # QKI -> Red channel
    channel = img[:, :, qki_channel]
    p2, p98 = np.percentile(channel, (2, 98))
    normalized = np.clip((channel - p2) / (p98 - p2), 0, 1).astype(np.float32)
    composite[:, :, 0] = (normalized * 255).astype(np.uint8)  # Red component
    
    # DAPI -> Blue channel
    channel = img[:, :, dapi_channel]
    p2, p98 = np.percentile(channel, (2, 98))
    normalized = np.clip((channel - p2) / (p98 - p2), 0, 1).astype(np.float32)
    composite[:, :, 2] = (normalized * 255).astype(np.uint8)  # Blue component
    
    # IBA1 -> Teal (mix of green and blue)
    channel = img[:, :, iba1_channel]
    p2, p98 = np.percentile(channel, (2, 98))
    normalized = np.clip((channel - p2) / (p98 - p2), 0, 1).astype(np.float32)
    composite[:, :, 1] = (normalized * 255).astype(np.uint8)  # Green component
    # Add some blue to make it more teal-like
    composite[:, :, 2] += (normalized * 128).astype(np.uint8)  # Add blue component
    
    # NEW: T cells -> Yellow (mix of red and green)
    if tcell_channel < img.shape[2]:
        channel = img[:, :, tcell_channel]
        p2, p98 = np.percentile(channel, (2, 98))
        normalized = np.clip((channel - p2) / (p98 - p2), 0, 1).astype(np.float32)
        
        # Add to both red and green channels to create yellow
        composite[:, :, 0] += (normalized * 128).astype(np.uint8)  # Red component
        composite[:, :, 1] += (normalized * 128).astype(np.uint8)  # Green component
    
    # Add IBA1 boundaries in red
    iba1_boundaries = segmentation.find_boundaries(iba1_mask, mode='outer', background=0)
    iba1_boundaries = morphology.binary_dilation(iba1_boundaries, morphology.disk(boundary_thickness))
    composite[iba1_boundaries, 0] = 255      # Red channel = 255
    composite[iba1_boundaries, 1] = 0        # Green channel = 0
    composite[iba1_boundaries, 2] = 0        # Blue channel = 0
    
    # Add nuclei in IBA1 boundaries in magenta
    nuclei_in_iba1_boundaries = segmentation.find_boundaries(nuclei_in_iba1_mask, mode='outer', background=0)
    nuclei_in_iba1_boundaries = morphology.binary_dilation(nuclei_in_iba1_boundaries, morphology.disk(boundary_thickness))
    composite[nuclei_in_iba1_boundaries, 0] = 255  # Red channel = 255
    composite[nuclei_in_iba1_boundaries, 1] = 0    # Green channel = 0
    composite[nuclei_in_iba1_boundaries, 2] = 255  # Blue channel = 255
    
    # NEW: Add T cell boundaries in yellow/orange
    if tcell_mask is not None and np.any(tcell_mask):
        tcell_boundaries = segmentation.find_boundaries(tcell_mask, mode='outer', background=0)
        tcell_boundaries = morphology.binary_dilation(tcell_boundaries, morphology.disk(boundary_thickness))
        composite[tcell_boundaries, 0] = 255  # Red channel = 255
        composite[tcell_boundaries, 1] = 165  # Green channel = 165
        composite[tcell_boundaries, 2] = 0    # Blue channel = 0
    
    # Create labeled image - directly use full resolution
    labels_rgb = label2rgb(nuclei_in_iba1_labels, bg_label=0, bg_color=(0, 0, 0))
    
    # Create IBA1 overlay - directly use full resolution
    iba1_overlay = label2rgb(iba1_labels, bg_label=0, bg_color=(0, 0, 0))
    
    # NEW: Create T cell overlay if available
    tcell_overlay = None
    if tcell_labels is not None and np.any(tcell_labels):
        tcell_overlay = label2rgb(tcell_labels, bg_label=0, bg_color=(0, 0, 0))
    
    # Create Qki visualization if we have data for it
    qki_vis = None
    if img.shape[2] >= 3:  # We need at least 3 channels
        # Extract Qki channel (assuming channel 0)
        qki_img = img[:, :, qki_channel]
        
        # Enhance contrast for visualization
        p2, p98 = np.percentile(qki_img, (2, 98))
        qki_enhanced = np.clip((qki_img - p2) / (p98 - p2), 0, 1).astype(np.float32)
        
        # Create Qki visualization at full resolution
        qki_vis = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
        qki_vis[:,:,0] = qki_enhanced  # Red channel for Qki
        qki_vis[:,:,1][iba1_boundaries] = 1.0  # Green boundaries for IBA1
        qki_vis[:,:,2][nuclei_in_iba1_boundaries] = 1.0  # Blue boundaries for nuclei in IBA1
        
        # NEW: Add T cell boundaries to Qki visualization if available
        if tcell_mask is not None and np.any(tcell_mask):
            tcell_boundaries = segmentation.find_boundaries(tcell_mask, mode='outer', background=0)
            tcell_boundaries = morphology.binary_dilation(tcell_boundaries, morphology.disk(boundary_thickness))
            qki_vis[:,:,0][tcell_boundaries] = 1.0  # Red channel
            qki_vis[:,:,1][tcell_boundaries] = 0.65  # Green channel (65%)
            # Creating orange/yellow boundaries for T cells
    
    # NEW: Create a special visualization showing IBA1 and T cells together
    iba1_tcell_vis = None
    if tcell_mask is not None and np.any(tcell_mask):
        iba1_tcell_vis = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        
        # IBA1 cells in red
        iba1_tcell_vis[iba1_mask, 0] = 200  # Red
        
        # T cells in green
        iba1_tcell_vis[tcell_mask, 1] = 200  # Green
        
        # Overlapping regions in yellow
        overlap_mask = iba1_mask & tcell_mask
        iba1_tcell_vis[overlap_mask, 0] = 200  # Red
        iba1_tcell_vis[overlap_mask, 1] = 200  # Green
        
        # Add boundaries
        iba1_tcell_vis[iba1_boundaries, 0] = 255  # Red boundary
        iba1_tcell_vis[tcell_boundaries, 1] = 255  # Green boundary
    
    visualizations = {
        'composite': composite,
        'nuclei_in_iba1_labeled': labels_rgb,
        'iba1_labeled': iba1_overlay
    }
    
    # Add T cell overlay if available
    if tcell_overlay is not None:
        visualizations['tcell_labeled'] = tcell_overlay
    
    # Add IBA1-T cell visualization if available
    if iba1_tcell_vis is not None:
        visualizations['iba1_tcell_overlay'] = iba1_tcell_vis
    
    # Add morphology and Qki visualizations if available
    if morphology_vis is not None:
        visualizations['morphology'] = morphology_vis
    
    if qki_vis is not None:
        visualizations['qki'] = (qki_vis * 255).astype(np.uint8)
    
    return visualizations
def save_results_with_qki(results, visualizations, output_dir):
    """
    Save analysis results and visualizations to disk with optimized performance,
    now including T cell results
    
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
                        'Hyperramified': (200, 50, 50) # BGR Blue
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
    
    # NEW: T cell properties CSV if available
    tcell_props_list = []
    if 'tcell_props' in results and results['tcell_props']:
        # Convert region properties to dictionaries for DataFrame
        for prop in results['tcell_props']:
            if hasattr(prop, 'label'):  # It's a RegionProperties object
                tcell_props_list.append({
                    'label': prop.label,
                    'area': prop.area,
                    'perimeter': prop.perimeter,
                    'solidity': prop.solidity,
                    'eccentricity': prop.eccentricity,
                    'centroid_y': prop.centroid[0],
                    'centroid_x': prop.centroid[1]
                })
            else:  # It's already a dictionary
                tcell_props_list.append(prop)
                
        if tcell_props_list:
            tcell_df = pd.DataFrame(tcell_props_list)
            csv_tasks.append((tcell_df, os.path.join(output_dir, "tcell_properties.csv")))
    
    # NEW: Cell centers CSV for spatial analysis
    # Combine IBA1 and T cell centers into one dataframe
    centers_data = []
    
    # Add IBA1 cell centers
    for prop in results['iba1_props']:
        centers_data.append({
            'cell_type': 'iba1',
            'centroid_x': prop.centroid[1],
            'centroid_y': prop.centroid[0]
        })
    
    # Add T cell centers
    for tcell in tcell_props_list:
        centers_data.append({
            'cell_type': 'tcell',
            'centroid_x': tcell['centroid_x'],
            'centroid_y': tcell['centroid_y']
        })
    
    # Save centers data
    if centers_data:
        centers_df = pd.DataFrame(centers_data)
        csv_tasks.append((centers_df, os.path.join(output_dir, "cell_centers.csv")))
        
        # Also save separate CSVs for each cell type
        iba1_centers = centers_df[centers_df['cell_type'] == 'iba1']
        tcell_centers = centers_df[centers_df['cell_type'] == 'tcell']
        
        csv_tasks.append((iba1_centers, os.path.join(output_dir, "iba1_cell_centers.csv")))
        csv_tasks.append((tcell_centers, os.path.join(output_dir, "tcell_centers.csv")))
    
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
                (0, 102, 255),   # Rod - BGR for Orange (new)
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
    
    # NEW: Create a chart showing IBA1 and T cell counts
    if 'stats' in results:
        print("Creating cell count comparison chart...")
        stats = results['stats']
        
        # Extract cell counts
        iba1_count = stats['iba1_count']
        tcell_count = stats.get('tcell_count', 0)
        
        if iba1_count > 0 or tcell_count > 0:
            # Create a bar chart
            width, height = 800, 600
            bar_img = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Calculate bar heights
            max_count = max(iba1_count, tcell_count)
            if max_count > 0:
                iba1_height = int((iba1_count / max_count) * (height - 150))
                tcell_height = int((tcell_count / max_count) * (height - 150))
                
                # Draw bars
                # IBA1 cell bar (teal)
                cv2.rectangle(bar_img, 
                             (200, height - 100 - iba1_height), 
                             (300, height - 100), 
                             (100, 180, 180),  # Teal in BGR
                             -1)
                
                # T cell bar (yellow)
                cv2.rectangle(bar_img, 
                             (500, height - 100 - tcell_height), 
                             (600, height - 100), 
                             (0, 200, 200),  # Yellow in BGR
                             -1)
                
                # Draw cell counts
                cv2.putText(bar_img, str(iba1_count), (200, height - 100 - iba1_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                cv2.putText(bar_img, str(tcell_count), (500, height - 100 - tcell_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # Draw x-axis labels
                cv2.putText(bar_img, "IBA1+ Cells", (170, height - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                cv2.putText(bar_img, "T Cells", (500, height - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # Draw title
                cv2.putText(bar_img, "Cell Count Comparison", (width // 2 - 150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
                
                # Save the chart
                cv2.imwrite(os.path.join(output_dir, "cell_count_comparison.png"), bar_img)
                print("  - Saved cell count comparison chart")
    
    # If available in the results, also save the correlation plots
    if 'morphology_classes' in results and results.get('nuclei_properties'):
        print("Creating and saving correlation plots...")
        save_correlation_plots(results, output_dir)
    
    # NEW: Generate comprehensive unified cell report
    save_spatial_analysis_csv(results, output_dir)
    unified_df = save_unified_cell_report(results, output_dir)
    total_time = time.time() - start_time
    print(f"Results saved to {output_dir} in {total_time:.2f} seconds")


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
            
            # NEW: Calculate additional metrics for artifact detection
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
    
    # NEW: Set artifact detection thresholds
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
        'artifact': 5,        # New class for artifacts
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
    class_masks = {}
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

def save_correlation_plots(results, output_dir):
    """
    Creates and saves correlation plots between morphology and Qki expression
    using OpenCV to maintain full resolution.
    Now also includes T cell vs IBA1 density correlation.
    
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
                'cell_type': 'iba1',  # Add cell type
                'morphology_class': morph_data['classification'],
                'area': prop.area,
                'perimeter': prop.perimeter,
                'solidity': prop.solidity,
                'eccentricity': prop.eccentricity,
                'centroid_y': y,
                'centroid_x': x
            })
        
        # Add T cell data if available
        if 'tcell_props' in results and results['tcell_props']:
            for prop in results['tcell_props']:
                prop_dict = {
                    'cell_id': prop.label if hasattr(prop, 'label') else 0,
                    'cell_type': 'tcell',
                    'morphology_class': 'tcell',
                    'area': prop.area if hasattr(prop, 'area') else prop.get('area', 0),
                    'perimeter': prop.perimeter if hasattr(prop, 'perimeter') else prop.get('perimeter', 0),
                    'solidity': prop.solidity if hasattr(prop, 'solidity') else prop.get('solidity', 0),
                    'eccentricity': prop.eccentricity if hasattr(prop, 'eccentricity') else prop.get('eccentricity', 0),
                    'centroid_y': prop.centroid[0] if hasattr(prop, 'centroid') else prop.get('centroid_y', 0),
                    'centroid_x': prop.centroid[1] if hasattr(prop, 'centroid') else prop.get('centroid_x', 0)
                }
                cell_data.append(prop_dict)
        
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
    
    # NEW: Create a spatial density analysis of IBA1 and T cells
    if 'cell_type' in cell_df.columns and 'centroid_x' in cell_df.columns:
        print("  - Creating spatial correlation analysis...")
        
        # Get IBA1 and T cell coordinates
        iba1_cells = cell_df[cell_df['cell_type'] == 'iba1']
        tcells = cell_df[cell_df['cell_type'] == 'tcell']
        
        if not iba1_cells.empty and not tcells.empty:
            # Create an image to analyze spatial correlation
            # Get image dimensions from results
            if 'image' in results:
                img_height, img_width = results['image'].shape[:2]
            else:
                # Use maximum coordinates as image dimensions
                img_width = int(cell_df['centroid_x'].max()) + 100
                img_height = int(cell_df['centroid_y'].max()) + 100
            
            # Create blank images
            iba1_density = np.zeros((img_height, img_width), dtype=np.uint8)
            tcell_density = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # Draw cells as points with gaussian spread
            for _, row in iba1_cells.iterrows():
                x, y = int(row['centroid_x']), int(row['centroid_y'])
                if 0 <= x < img_width and 0 <= y < img_height:
                    cv2.circle(iba1_density, (x, y), 3, 255, -1)
            
            for _, row in tcells.iterrows():
                x, y = int(row['centroid_x']), int(row['centroid_y'])
                if 0 <= x < img_width and 0 <= y < img_height:
                    cv2.circle(tcell_density, (x, y), 3, 255, -1)
            
            # Blur the density maps to create heatmaps
            iba1_heatmap = cv2.GaussianBlur(iba1_density, (151, 151), 30)
            tcell_heatmap = cv2.GaussianBlur(tcell_density, (151, 151), 30)
            
            # Normalize the heatmaps for better visualization
            iba1_heatmap_norm = cv2.normalize(iba1_heatmap, None, 0, 255, cv2.NORM_MINMAX)
            tcell_heatmap_norm = cv2.normalize(tcell_heatmap, None, 0, 255, cv2.NORM_MINMAX)
            
            # Create RGB heatmap images
            iba1_heat_rgb = cv2.applyColorMap(iba1_heatmap_norm, cv2.COLORMAP_JET)
            tcell_heat_rgb = cv2.applyColorMap(tcell_heatmap_norm, cv2.COLORMAP_JET)
            
            # Create a combined visualization
            combined_heatmap = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            # IBA1 in red channel
            combined_heatmap[:,:,2] = iba1_heatmap_norm  # Red channel
            # T cells in green channel
            combined_heatmap[:,:,1] = tcell_heatmap_norm  # Green channel
            
            # Save the heatmaps
            cv2.imwrite(os.path.join(output_dir, "iba1_density_heatmap.png"), iba1_heat_rgb)
            cv2.imwrite(os.path.join(output_dir, "tcell_density_heatmap.png"), tcell_heat_rgb)
            cv2.imwrite(os.path.join(output_dir, "combined_density_heatmap.png"), combined_heatmap)
            print("  - Saved cell density heatmaps")
            
            # Basic co-localization analysis
            # Calculate 2D correlation coefficient between heatmaps
            try:
                correlation = np.corrcoef(iba1_heatmap.flatten(), tcell_heatmap.flatten())[0, 1]
                
                # Save correlation result
                with open(os.path.join(output_dir, "spatial_correlation.txt"), 'w') as f:
                    f.write(f"Spatial correlation between IBA1+ cells and T cells: {correlation:.4f}\n")
                    f.write("\nPositive values indicate regions where both cell types tend to be found together.\n")
                    f.write("Negative values indicate regions where the presence of one cell type correlates with the absence of the other.\n")
                    f.write("Values close to zero indicate little to no spatial relationship.\n")
                
                print(f"  - Spatial correlation: {correlation:.4f}")
            except Exception as e:
                print(f"  - Error calculating spatial correlation: {e}")

def analyze_cell_distributions(results, output_dir):
    """
    NEW FUNCTION: Analyze spatial distribution and interactions between IBA1 cells and T cells
    
    Parameters:
    -----------
    results : dict
        Results dictionary containing cell data
    output_dir : str
        Directory to save analysis results
    
    Returns:
    --------
    dict
        Dictionary containing distribution analysis results
    """
    # Extract cell data
    iba1_props = results['iba1_props']
    tcell_props = results.get('tcell_props', [])
    
    # Skip if not enough data
    if len(iba1_props) == 0 or len(tcell_props) == 0:
        print("Not enough cell data for distribution analysis")
        return None
    
    print("Analyzing cell spatial distributions...")
    
    # Collect cell coordinates
    iba1_coords = []
    for prop in iba1_props:
        iba1_coords.append((prop.centroid[1], prop.centroid[0]))  # x, y format
    
    tcell_coords = []
    for prop in tcell_props:
        if hasattr(prop, 'centroid'):
            tcell_coords.append((prop.centroid[1], prop.centroid[0]))  # x, y format
        else:  # Dictionary format
            tcell_coords.append((prop.get('centroid_x', 0), prop.get('centroid_y', 0)))
    
    iba1_coords = np.array(iba1_coords)
    tcell_coords = np.array(tcell_coords)
    
    # Calculate nearest neighbor distances
    iba1_to_iba1_distances = []
    iba1_to_tcell_distances = []
    tcell_to_iba1_distances = []
    tcell_to_tcell_distances = []
    
    # For each IBA1 cell, find nearest IBA1 and T cell
    for i, iba1_coord in enumerate(iba1_coords):
        # Find nearest IBA1 (excluding self)
        iba1_distances = np.sqrt(np.sum((iba1_coords - iba1_coord) ** 2, axis=1))
        iba1_distances[i] = np.inf  # Exclude self
        iba1_to_iba1_distances.append(np.min(iba1_distances))
        
        # Find nearest T cell
        tcell_distances = np.sqrt(np.sum((tcell_coords - iba1_coord) ** 2, axis=1))
        iba1_to_tcell_distances.append(np.min(tcell_distances))
    
    # For each T cell, find nearest IBA1 and T cell
    for i, tcell_coord in enumerate(tcell_coords):
        # Find nearest IBA1
        iba1_distances = np.sqrt(np.sum((iba1_coords - tcell_coord) ** 2, axis=1))
        tcell_to_iba1_distances.append(np.min(iba1_distances))
        
        # Find nearest T cell (excluding self)
        tcell_distances = np.sqrt(np.sum((tcell_coords - tcell_coord) ** 2, axis=1))
        tcell_distances[i] = np.inf  # Exclude self
        if len(tcell_distances) > 1:  # Need at least two T cells
            tcell_to_tcell_distances.append(np.min(tcell_distances))
    
    # Calculate statistics
    iba1_iba1_mean = np.mean(iba1_to_iba1_distances)
    iba1_tcell_mean = np.mean(iba1_to_tcell_distances)
    tcell_iba1_mean = np.mean(tcell_to_iba1_distances)
    tcell_tcell_mean = np.mean(tcell_to_tcell_distances) if tcell_to_tcell_distances else 0
    
    # Visualize distributions with histograms
    plt.figure(figsize=(12, 10))
    
    # IBA1-IBA1 distances
    plt.subplot(2, 2, 1)
    plt.hist(iba1_to_iba1_distances, bins=20, alpha=0.7, color='teal')
    plt.axvline(iba1_iba1_mean, color='red', linestyle='--')
    plt.title('IBA1 to nearest IBA1 distances')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Count')
    
    # IBA1-T cell distances
    plt.subplot(2, 2, 2)
    plt.hist(iba1_to_tcell_distances, bins=20, alpha=0.7, color='orange')
    plt.axvline(iba1_tcell_mean, color='red', linestyle='--')
    plt.title('IBA1 to nearest T cell distances')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Count')
    
    # T cell-IBA1 distances
    plt.subplot(2, 2, 3)
    plt.hist(tcell_to_iba1_distances, bins=20, alpha=0.7, color='purple')
    plt.axvline(tcell_iba1_mean, color='red', linestyle='--')
    plt.title('T cell to nearest IBA1 distances')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Count')
    
    # T cell-T cell distances
    if tcell_to_tcell_distances:
        plt.subplot(2, 2, 4)
        plt.hist(tcell_to_tcell_distances, bins=20, alpha=0.7, color='green')
        plt.axvline(tcell_tcell_mean, color='red', linestyle='--')
        plt.title('T cell to nearest T cell distances')
        plt.xlabel('Distance (pixels)')
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cell_distance_distributions.png"), dpi=300)
    plt.close()
    
    # Create spatial point map
    plt.figure(figsize=(10, 10))
    plt.scatter(iba1_coords[:, 0], iba1_coords[:, 1], c='teal', alpha=0.7, label='IBA1+ Cells')
    plt.scatter(tcell_coords[:, 0], tcell_coords[:, 1], c='orange', alpha=0.7, label='T Cells')
    plt.title('Spatial Distribution of Cells')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "cell_spatial_map.png"), dpi=300)
    plt.close()
    
    # Save quantitative results to CSV
    distance_data = pd.DataFrame({
        'iba1_to_nearest_iba1': iba1_to_iba1_distances,
        'iba1_to_nearest_tcell': iba1_to_tcell_distances
    })
    distance_data.to_csv(os.path.join(output_dir, "iba1_cell_distances.csv"), index=False)
    
    tcell_distance_data = pd.DataFrame({
        'tcell_to_nearest_iba1': tcell_to_iba1_distances,
        'tcell_to_nearest_tcell': tcell_to_tcell_distances if len(tcell_to_tcell_distances) > 0 else [0] * len(tcell_to_iba1_distances)
    })
    tcell_distance_data.to_csv(os.path.join(output_dir, "tcell_distances.csv"), index=False)
    
    # Define close interaction threshold (in pixels)
    close_interaction_threshold = 50  # Adjust based on your data scale
    
    # Count cells in close proximity
    iba1_near_tcell_count = sum(d < close_interaction_threshold for d in iba1_to_tcell_distances)
    tcell_near_iba1_count = sum(d < close_interaction_threshold for d in tcell_to_iba1_distances)
    
    iba1_near_tcell_percent = (iba1_near_tcell_count / len(iba1_props)) * 100 if iba1_props else 0
    tcell_near_iba1_percent = (tcell_near_iba1_count / len(tcell_props)) * 100 if tcell_props else 0
    
    # Create interaction summary
    summary_data = {
        'mean_iba1_to_iba1_distance': iba1_iba1_mean,
        'mean_iba1_to_tcell_distance': iba1_tcell_mean,
        'mean_tcell_to_iba1_distance': tcell_iba1_mean,
        'mean_tcell_to_tcell_distance': tcell_tcell_mean,
        'iba1_cells_near_tcell_count': iba1_near_tcell_count,
        'iba1_cells_near_tcell_percent': iba1_near_tcell_percent,
        'tcell_near_iba1_count': tcell_near_iba1_count,
        'tcell_near_iba1_percent': tcell_near_iba1_percent,
        'interaction_threshold_pixels': close_interaction_threshold
    }
    
    # Save summary to CSV
    pd.DataFrame([summary_data]).to_csv(os.path.join(output_dir, "cell_interaction_summary.csv"), index=False)
    
    # Create bar chart of interaction percentages
    plt.figure(figsize=(8, 6))
    plt.bar(['IBA1+ cells near T cells', 'T cells near IBA1+ cells'], 
            [iba1_near_tcell_percent, tcell_near_iba1_percent], 
            color=['teal', 'orange'])
    plt.title(f'Cell Interactions (threshold: {close_interaction_threshold} pixels)')
    plt.ylabel('Percentage of cells (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "cell_interaction_percentages.png"), dpi=300)
    plt.close()
    
    print(f"  - {iba1_near_tcell_percent:.1f}% of IBA1+ cells are near T cells")
    print(f"  - {tcell_near_iba1_percent:.1f}% of T cells are near IBA1+ cells")
    print("Cell distribution analysis completed")
    
    return summary_data


def main(image_path, output_dir, 
         iba1_channel=1, dapi_channel=2, qki_channel=0, tcell_channel=3,
         iba1_min_size=50, iba1_max_size=5000,
         dapi_min_size=15, dapi_max_size=500,
         tcell_min_size=20, tcell_max_size=2000,
         iba1_threshold_method="adaptive", dapi_threshold_method="otsu",
         use_enhanced_iba1=True, use_enhanced_tcell=True,
         remove_artifacts=True, artifact_area_threshold=8000,
         iba1_min_intensity=None, iba1_max_intensity=None,
         tcell_min_intensity=None, tcell_max_intensity=None,
         tile_size=2000, overlap=200, max_workers=4):
    """
    Main function to run the entire analysis pipeline for IBA1+, T cells, DAPI+ nuclei and Qki expression
    
    Parameters:
    -----------
    image_path : str
        Path to the multi-channel image file
    output_dir : str
        Directory to save the results
    iba1_channel, dapi_channel, qki_channel, tcell_channel : int
        Channel indices for staining (0-indexed)
    iba1_min_size, iba1_max_size, dapi_min_size, dapi_max_size, tcell_min_size, tcell_max_size : int
        Size filters for cells and nuclei in pixels
    iba1_threshold_method, dapi_threshold_method : str
        Methods for thresholding
    use_enhanced_iba1, use_enhanced_tcell : bool
        Whether to use the enhanced detection algorithms
    remove_artifacts : bool
        Whether to apply additional artifact removal steps
    artifact_area_threshold : int
        Size threshold above which objects are considered potential artifacts
    iba1_min_intensity, iba1_max_intensity, tcell_min_intensity, tcell_max_intensity : float or None
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
        f.write(f"Channels: IBA1={iba1_channel}, DAPI={dapi_channel}, QKI={qki_channel}, T cell={tcell_channel}\n")
        f.write(f"Size filters: IBA1={iba1_min_size}-{iba1_max_size}, DAPI={dapi_min_size}-{dapi_max_size}, T cell={tcell_min_size}-{tcell_max_size}\n")
        f.write(f"Threshold methods: IBA1={iba1_threshold_method}, DAPI={dapi_threshold_method}\n")
        f.write(f"Enhanced detection: IBA1={use_enhanced_iba1}, T cell={use_enhanced_tcell}\n")
        f.write(f"Artifact removal: {remove_artifacts}, threshold={artifact_area_threshold}\n")
        f.write(f"Intensity filters: IBA1={iba1_min_intensity}-{iba1_max_intensity}, T cell={tcell_min_intensity}-{tcell_max_intensity}\n")
        f.write(f"Tiling parameters: size={tile_size}, overlap={overlap}, workers={max_workers}\n")
        f.write(f"Analysis started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Run cell detection and analysis
    results = detect_nuclei_inside_iba1_with_qki_tiled(
        image_path=image_path,
        iba1_channel=iba1_channel,
        dapi_channel=dapi_channel,
        qki_channel=qki_channel,
        tcell_channel=tcell_channel,
        iba1_threshold_method=iba1_threshold_method,
        dapi_threshold_method=dapi_threshold_method,
        iba1_min_size=iba1_min_size,
        iba1_max_size=iba1_max_size,
        dapi_min_size=dapi_min_size,
        dapi_max_size=dapi_max_size,
        tcell_min_size=tcell_min_size,
        tcell_max_size=tcell_max_size,
        boundary_thickness=2,
        cytoplasm_expansion=3,
        tile_size=tile_size,
        overlap=overlap,
        max_workers=max_workers,
        use_enhanced_iba1=use_enhanced_iba1,
        use_enhanced_tcell=use_enhanced_tcell,
        remove_artifacts=remove_artifacts,
        artifact_area_threshold=artifact_area_threshold,
        artifact_solidity_threshold=0.95,
        iba1_min_intensity=iba1_min_intensity,
        iba1_max_intensity=iba1_max_intensity,
        tcell_min_intensity=tcell_min_intensity,
        tcell_max_intensity=tcell_max_intensity
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


    
    # NEW: Generate spatial analysis CSV files
    print("Generating spatial analysis CSV files...")
    save_spatial_analysis_csv(results, output_dir)
    
    # Generate unified comprehensive cell report with all metrics in one CSV
    print("Generating unified cell report...")
    unified_df = save_unified_cell_report(results, output_dir)

    
    # NEW: Perform spatial distribution analysis of IBA1 and T cells
    if 'tcell_props' in results and results['tcell_props'] and len(results['tcell_props']) > 0:
        print("Performing spatial distribution analysis...")
        distribution_summary = analyze_cell_distributions(results, output_dir)
        if distribution_summary:
            # Add to results
            results['distribution_summary'] = distribution_summary
    
    # Print final summary
    total_time = time.time() - start_time
    print("\nFinal Summary:")
    print(f"IBA1+ cells: {results['stats']['iba1_count']}")
    print(f"DAPI+ nuclei: {results['stats']['nuclei_count']}")
    print(f"T cells: {results['stats']['tcell_count']}")
    print(f"Analysis completed in {total_time/60:.1f} minutes")
    
    return results

def create_iba1_tcell_plots(output_dir, results_files):
    """
    Create comparative plots from multiple analysis results to compare IBA1 and T cell metrics
    across different images.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the plots
    results_files : list
        List of paths to result directories for different samples
        
    Returns:
    --------
    None
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data from all results
    all_data = []
    
    for result_dir in results_files:
        # Check if it's a directory
        if not os.path.isdir(result_dir):
            continue
            
        # Try to load the stats CSV
        stats_path = os.path.join(result_dir, "stats.csv")
        if os.path.exists(stats_path):
            stats_df = pd.read_csv(stats_path)
            if not stats_df.empty:
                # Extract sample name from directory
                sample_name = os.path.basename(result_dir)
                
                # Add sample name to stats
                stats_row = stats_df.iloc[0].to_dict()
                stats_row['sample'] = sample_name
                
                # Check for required columns
                required_cols = ['iba1_count', 'tcell_count', 'avg_nuclear_qki']
                if all(col in stats_row for col in required_cols):
                    all_data.append(stats_row)
    
    # If we have data, create plots
    if all_data:
        df = pd.DataFrame(all_data)
        
        # 1. Cell count comparison across samples
        plt.figure(figsize=(12, 6))
        
        samples = df['sample'].tolist()
        iba1_counts = df['iba1_count'].tolist()
        tcell_counts = df['tcell_count'].tolist()
        
        x = np.arange(len(samples))
        width = 0.35
        
        plt.bar(x - width/2, iba1_counts, width, label='IBA1+ Cells', color='teal')
        plt.bar(x + width/2, tcell_counts, width, label='T Cells', color='orange')
        
        plt.xlabel('Sample')
        plt.ylabel('Cell Count')
        plt.title('IBA1+ and T Cell Counts Across Samples')
        plt.xticks(x, samples, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cell_count_comparison_all_samples.png"), dpi=300)
        plt.close()
        
        # 2. Cell ratio comparison
        plt.figure(figsize=(10, 6))
        
        # Calculate ratio
        df['tcell_to_iba1_ratio'] = df['tcell_count'] / df['iba1_count']
        
        plt.bar(samples, df['tcell_to_iba1_ratio'], color='purple')
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        plt.xlabel('Sample')
        plt.ylabel('T Cell : IBA1+ Cell Ratio')
        plt.title('Ratio of T Cells to IBA1+ Cells')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tcell_to_iba1_ratio.png"), dpi=300)
        plt.close()
        
        # 3. QKI expression comparison
        if 'avg_nuclear_qki' in df.columns and 'avg_cytoplasmic_qki' in df.columns and 'avg_nc_ratio' in df.columns:
            plt.figure(figsize=(12, 6))
            
            nuclear_qki = df['avg_nuclear_qki'].tolist()
            cyto_qki = df['avg_cytoplasmic_qki'].tolist()
            
            x = np.arange(len(samples))
            width = 0.35
            
            plt.bar(x - width/2, nuclear_qki, width, label='Nuclear QKI', color='blue')
            plt.bar(x + width/2, cyto_qki, width, label='Cytoplasmic QKI', color='green')
            
            plt.xlabel('Sample')
            plt.ylabel('QKI Intensity')
            plt.title('QKI Expression in IBA1+ Cells Across Samples')
            plt.xticks(x, samples, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "qki_expression_comparison.png"), dpi=300)
            plt.close()
            
            # 4. N:C Ratio
            plt.figure(figsize=(10, 6))
            
            plt.bar(samples, df['avg_nc_ratio'], color='red')
            
            plt.xlabel('Sample')
            plt.ylabel('N:C Ratio')
            plt.title('Nuclear:Cytoplasmic QKI Ratio in IBA1+ Cells')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "nc_ratio_comparison.png"), dpi=300)
            plt.close()
        
        # 5. Morphology comparison if available
        morphology_columns = [col for col in df.columns if col.endswith('_cells') and col != 'iba1_count' and col != 'tcell_count']
        if morphology_columns:
            plt.figure(figsize=(14, 8))
            
            # Prepare data for stacked bar chart
            bottom = np.zeros(len(samples))
            
            for col in morphology_columns:
                morph_class = col.replace('_cells', '')
                if morph_class in ['ameboid', 'ramified', 'hyperramified', 'rod']:
                    values = df[col].tolist()
                    
                    # Choose color based on class
                    if morph_class == 'ameboid':
                        color = 'red'
                    elif morph_class == 'ramified':
                        color = 'green'
                    elif morph_class == 'hyperramified':
                        color = 'blue'
                    elif morph_class == 'rod':
                        color = 'orange'
                    else:
                        color = 'gray'
                    
                    plt.bar(samples, values, bottom=bottom, label=morph_class.capitalize(), color=color)
                    bottom += np.array(values)
            
            plt.xlabel('Sample')
            plt.ylabel('Cell Count')
            plt.title('IBA1+ Cell Morphology Distribution Across Samples')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "morphology_comparison.png"), dpi=300)
            plt.close()
            
            # 6. Morphology percentage comparison
            plt.figure(figsize=(14, 8))
            
            # Calculate percentages
            for col in morphology_columns:
                if col in ['ameboid_cells', 'ramified_cells', 'hyperramified_cells', 'rod_cells']:
                    morph_class = col.replace('_cells', '')
                    df[f'{morph_class}_percent'] = df[col] / df['iba1_count'] * 100
            
            # Plot percentage stacked bar chart
            bottom = np.zeros(len(samples))
            
            for morph_class in ['ameboid', 'ramified', 'hyperramified', 'rod']:
                col = f'{morph_class}_percent'
                if col in df.columns:
                    values = df[col].tolist()
                    
                    # Choose color based on class
                    if morph_class == 'ameboid':
                        color = 'red'
                    elif morph_class == 'ramified':
                        color = 'green'
                    elif morph_class == 'hyperramified':
                        color = 'blue'
                    elif morph_class == 'rod':
                        color = 'orange'
                    else:
                        color = 'gray'
                    
                    plt.bar(samples, values, bottom=bottom, label=morph_class.capitalize(), color=color)
                    bottom += np.array(values)
            
            plt.xlabel('Sample')
            plt.ylabel('Percentage (%)')
            plt.title('IBA1+ Cell Morphology Distribution Percentage Across Samples')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "morphology_percentage_comparison.png"), dpi=300)
            plt.close()
        
        # Save combined dataframe
        df.to_csv(os.path.join(output_dir, "all_samples_summary.csv"), index=False)
        print(f"Created comparative plots for {len(samples)} samples")
    else:
        print("Not enough data for comparative plots")
        
def save_spatial_analysis_csv(results, output_dir, pixel_size_um=0.35):
    """
    Saves CSV files with spatial analysis between macrophages (IBA1) and T cells.
    Creates two analysis approaches with detailed morphology features and distance conversions:
    1. Macrophage-centered: Each macrophage with its morphology features and distance to closest T cell
    2. T cell-centered: Each T cell with morphology features of 1st to 5th closest macrophages
    
    Parameters:
    -----------
    results : dict
        Results dictionary from detect_nuclei_inside_iba1_with_qki_tiled
    output_dir : str
        Directory to save the CSV files
    pixel_size_um : float
        Pixel size in micrometers (default: 0.35 for your image)
    
    Returns:
    --------
    None
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import cdist
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating enhanced spatial analysis CSV files...")
    print(f"Using pixel size: {pixel_size_um} μm/pixel")
    
    # Extract IBA1 (macrophage) data
    iba1_props = results['iba1_props']
    morphology_classes = results.get('morphology_classes', {})
    
    # Extract T cell data
    tcell_props = results.get('tcell_props', [])
    
    if len(iba1_props) == 0 or len(tcell_props) == 0:
        print("Insufficient cell data for spatial analysis")
        return
    
    # Prepare macrophage data with detailed morphology features
    macrophage_data = []
    macrophage_coords = []
    
    for prop in iba1_props:
        cell_label = prop.label
        y, x = prop.centroid
        
        # Get morphology classification and detailed metrics
        morphology_class = 'unknown'
        detailed_metrics = {}
        
        if morphology_classes and cell_label in morphology_classes:
            morph_info = morphology_classes[cell_label]
            morphology_class = morph_info['classification']
            # Get detailed metrics if available
            if 'metrics' in morph_info:
                detailed_metrics = morph_info['metrics']
        
        # Calculate additional morphological features
        area = prop.area
        perimeter = prop.perimeter
        solidity = prop.solidity
        eccentricity = prop.eccentricity
        
        # Calculate additional derived metrics
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        complexity = perimeter / np.sqrt(area) if area > 0 else 0
        aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 0
        roundness = prop.minor_axis_length / prop.major_axis_length if prop.major_axis_length > 0 else 0
        extent = area / (prop.bbox[2] - prop.bbox[0]) / (prop.bbox[3] - prop.bbox[1]) if (prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1]) > 0 else 0
        
        # Convert area and perimeter to micrometers
        area_um2 = area * (pixel_size_um ** 2)  # Area in μm²
        perimeter_um = perimeter * pixel_size_um  # Perimeter in μm
        major_axis_um = prop.major_axis_length * pixel_size_um
        minor_axis_um = prop.minor_axis_length * pixel_size_um
        
        macrophage_info = {
            'macrophage_id': cell_label,
            'centroid_x': x,
            'centroid_y': y,
            'morphology_classification': morphology_class,
            # Basic morphology features (pixels)
            'area_pixels': area,
            'perimeter_pixels': perimeter,
            'major_axis_length_pixels': prop.major_axis_length,
            'minor_axis_length_pixels': prop.minor_axis_length,
            # Basic morphology features (micrometers)
            'area_um2': area_um2,
            'perimeter_um': perimeter_um,
            'major_axis_length_um': major_axis_um,
            'minor_axis_length_um': minor_axis_um,
            # Shape descriptors (dimensionless)
            'solidity': solidity,
            'eccentricity': eccentricity,
            'circularity': circularity,
            'complexity': complexity,
            'aspect_ratio': aspect_ratio,
            'roundness': roundness,
            'extent': extent
        }
        
        # Add detailed metrics from morphology classification if available
        if detailed_metrics:
            for key, value in detailed_metrics.items():
                if key not in ['label', 'centroid_x', 'centroid_y']:  # Avoid duplicates
                    macrophage_info[f'morph_{key}'] = value
        
        macrophage_data.append(macrophage_info)
        macrophage_coords.append([x, y])
    
    # Prepare T cell data with detailed features
    tcell_data = []
    tcell_coords = []
    
    for i, prop in enumerate(tcell_props):
        if hasattr(prop, 'label'):
            tcell_id = prop.label
            y, x = prop.centroid
            area = prop.area
            perimeter = prop.perimeter
            solidity = prop.solidity
            eccentricity = prop.eccentricity
            major_axis = prop.major_axis_length
            minor_axis = prop.minor_axis_length
        else:  # Dictionary format
            tcell_id = prop.get('label', i + 1)
            x = prop.get('centroid_x', 0)
            y = prop.get('centroid_y', 0)
            area = prop.get('area', 0)
            perimeter = prop.get('perimeter', 0)
            solidity = prop.get('solidity', 0)
            eccentricity = prop.get('eccentricity', 0)
            major_axis = prop.get('major_axis_length', 0)
            minor_axis = prop.get('minor_axis_length', 0)
        
        # Calculate additional metrics for T cells
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
        roundness = minor_axis / major_axis if major_axis > 0 else 0
        
        # Convert to micrometers
        area_um2 = area * (pixel_size_um ** 2)
        perimeter_um = perimeter * pixel_size_um
        major_axis_um = major_axis * pixel_size_um
        minor_axis_um = minor_axis * pixel_size_um
        
        tcell_info = {
            'tcell_id': tcell_id,
            'centroid_x': x,
            'centroid_y': y,
            # T cell morphology features (pixels)
            'area_pixels': area,
            'perimeter_pixels': perimeter,
            'major_axis_length_pixels': major_axis,
            'minor_axis_length_pixels': minor_axis,
            # T cell morphology features (micrometers)
            'area_um2': area_um2,
            'perimeter_um': perimeter_um,
            'major_axis_length_um': major_axis_um,
            'minor_axis_length_um': minor_axis_um,
            # T cell shape descriptors
            'solidity': solidity,
            'eccentricity': eccentricity,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'roundness': roundness
        }
        
        tcell_data.append(tcell_info)
        tcell_coords.append([x, y])
    
    # Convert to numpy arrays for distance calculation
    macrophage_coords = np.array(macrophage_coords)
    tcell_coords = np.array(tcell_coords)
    
    # === MACROPHAGE-CENTERED ANALYSIS ===
    print("Processing macrophage-centered analysis with detailed features...")
    
    # Calculate distances from each macrophage to all T cells
    macro_to_tcell_distances = cdist(macrophage_coords, tcell_coords)
    
    macrophage_analysis = []
    for i, macro_info in enumerate(macrophage_data):
        # Find closest T cell
        distances_to_tcells = macro_to_tcell_distances[i]
        closest_tcell_idx = np.argmin(distances_to_tcells)
        closest_distance_pixels = distances_to_tcells[closest_tcell_idx]
        closest_distance_um = closest_distance_pixels * pixel_size_um
        
        # Create analysis row with all macrophage features
        analysis_row = macro_info.copy()  # Include all morphology features
        
        # Add nuclei information for this macrophage
        cell_label = macro_info['macrophage_id']
        iba1_labels = results['iba1_labels']
        nuclei_properties = results['nuclei_properties']
        
        # Find nuclei associated with this macrophage
        cell_nuclei = []
        if nuclei_properties:
            # Get macrophage position and approximate radius
            macro_y, macro_x = macro_info['centroid_y'], macro_info['centroid_x']
            cell_radius = np.sqrt(macro_info['area_pixels'] / np.pi)
            
            for nucleus in nuclei_properties:
                # Calculate distance from nucleus to macrophage centroid
                dist = np.sqrt((nucleus['y'] - macro_y)**2 + (nucleus['x'] - macro_x)**2)
                
                # Check if nucleus is within this macrophage (using distance threshold)
                if dist <= 1.5 * cell_radius:
                    cell_nuclei.append(nucleus)
        
        nuclei_count = len(cell_nuclei)
        has_nuclei = nuclei_count > 0
        
        # Add distance information and nuclei information
        analysis_row.update({
            'has_nuclei': has_nuclei,
            'nuclei_count': nuclei_count,
            'distance_to_closest_tcell_pixels': closest_distance_pixels,
            'distance_to_closest_tcell_um': closest_distance_um,
            'closest_tcell_id': tcell_data[closest_tcell_idx]['tcell_id']
        })
        
        # Add closest T cell features with prefix
        closest_tcell = tcell_data[closest_tcell_idx]
        for key, value in closest_tcell.items():
            if key not in ['tcell_id', 'centroid_x', 'centroid_y']:
                analysis_row[f'closest_tcell_{key}'] = value
        
        macrophage_analysis.append(analysis_row)
    
    # Save macrophage-centered CSV
    macro_df = pd.DataFrame(macrophage_analysis)
    macro_csv_path = os.path.join(output_dir, "macrophage_centered_analysis_detailed.csv")
    macro_df.to_csv(macro_csv_path, index=False)
    print(f"Detailed macrophage-centered analysis saved to: {macro_csv_path}")
    
    # === T CELL-CENTERED ANALYSIS ===
    print("Processing T cell-centered analysis with detailed features...")
    
    # Calculate distances from each T cell to all macrophages
    tcell_to_macro_distances = cdist(tcell_coords, macrophage_coords)
    
    tcell_analysis = []
    for i, tcell_info in enumerate(tcell_data):
        # Find 5 closest macrophages (or fewer if less than 5 exist)
        distances_to_macros = tcell_to_macro_distances[i]
        closest_macro_indices = np.argsort(distances_to_macros)[:5]  # Top 5 closest
        
        # Start with T cell's own features
        tcell_row = tcell_info.copy()
        
        # Add morphologies and detailed features of closest macrophages
        for j, macro_idx in enumerate(closest_macro_indices):
            if j < len(macrophage_data):  # Make sure index is valid
                macro_data = macrophage_data[macro_idx]
                distance_pixels = distances_to_macros[macro_idx]
                distance_um = distance_pixels * pixel_size_um
                
                # Add basic info for this closest macrophage
                tcell_row[f'closest_macro_{j+1}_id'] = macro_data['macrophage_id']
                tcell_row[f'closest_macro_{j+1}_morphology'] = macro_data['morphology_classification']
                tcell_row[f'closest_macro_{j+1}_distance_pixels'] = distance_pixels
                tcell_row[f'closest_macro_{j+1}_distance_um'] = distance_um
                
                # Add detailed morphology features for this macrophage
                feature_keys = ['area_pixels', 'area_um2', 'perimeter_pixels', 'perimeter_um',
                               'solidity', 'eccentricity', 'circularity', 'complexity', 
                               'aspect_ratio', 'roundness', 'extent']
                
                for key in feature_keys:
                    if key in macro_data:
                        tcell_row[f'closest_macro_{j+1}_{key}'] = macro_data[key]
            else:
                # Fill with N/A if not enough macrophages
                tcell_row[f'closest_macro_{j+1}_id'] = 'N/A'
                tcell_row[f'closest_macro_{j+1}_morphology'] = 'N/A'
                tcell_row[f'closest_macro_{j+1}_distance_pixels'] = np.inf
                tcell_row[f'closest_macro_{j+1}_distance_um'] = np.inf
                
                # Fill morphology features with N/A
                feature_keys = ['area_pixels', 'area_um2', 'perimeter_pixels', 'perimeter_um',
                               'solidity', 'eccentricity', 'circularity', 'complexity', 
                               'aspect_ratio', 'roundness', 'extent']
                
                for key in feature_keys:
                    tcell_row[f'closest_macro_{j+1}_{key}'] = np.nan
        
        # Fill remaining columns if less than 5 macrophages
        for j in range(len(closest_macro_indices), 5):
            tcell_row[f'closest_macro_{j+1}_id'] = 'N/A'
            tcell_row[f'closest_macro_{j+1}_morphology'] = 'N/A'
            tcell_row[f'closest_macro_{j+1}_distance_pixels'] = np.inf
            tcell_row[f'closest_macro_{j+1}_distance_um'] = np.inf
            
            feature_keys = ['area_pixels', 'area_um2', 'perimeter_pixels', 'perimeter_um',
                           'solidity', 'eccentricity', 'circularity', 'complexity', 
                           'aspect_ratio', 'roundness', 'extent']
            
            for key in feature_keys:
                tcell_row[f'closest_macro_{j+1}_{key}'] = np.nan
        
        tcell_analysis.append(tcell_row)
    
    # Save T cell-centered CSV
    tcell_df = pd.DataFrame(tcell_analysis)
    tcell_csv_path = os.path.join(output_dir, "tcell_centered_analysis_detailed.csv")
    tcell_df.to_csv(tcell_csv_path, index=False)
    print(f"Detailed T cell-centered analysis saved to: {tcell_csv_path}")
    
    # === ENHANCED SUMMARY ===
    print("Creating enhanced spatial analysis summary...")
    
    # Calculate summary statistics
    macro_distances_pixels = [row['distance_to_closest_tcell_pixels'] for row in macrophage_analysis]
    macro_distances_um = [row['distance_to_closest_tcell_um'] for row in macrophage_analysis]
    
    summary_stats = {
        'pixel_size_um_per_pixel': pixel_size_um,
        'total_macrophages': len(macrophage_data),
        'total_tcells': len(tcell_data),
        'avg_macro_to_tcell_distance_pixels': np.mean(macro_distances_pixels),
        'avg_macro_to_tcell_distance_um': np.mean(macro_distances_um),
        'median_macro_to_tcell_distance_pixels': np.median(macro_distances_pixels),
        'median_macro_to_tcell_distance_um': np.median(macro_distances_um),
        'min_macro_to_tcell_distance_pixels': np.min(macro_distances_pixels),
        'min_macro_to_tcell_distance_um': np.min(macro_distances_um),
        'max_macro_to_tcell_distance_pixels': np.max(macro_distances_pixels),
        'max_macro_to_tcell_distance_um': np.max(macro_distances_um)
    }
    
    # Add morphology breakdown with average features
    morphology_summary = {}
    for morph_class in ['ameboid', 'ramified', 'hyperramified', 'rod']:
        morph_macros = [m for m in macrophage_analysis if m['morphology_classification'] == morph_class]
        if morph_macros:
            morphology_summary[f'{morph_class}_count'] = len(morph_macros)
            morphology_summary[f'{morph_class}_avg_area_um2'] = np.mean([m['area_um2'] for m in morph_macros])
            morphology_summary[f'{morph_class}_avg_solidity'] = np.mean([m['solidity'] for m in morph_macros])
            morphology_summary[f'{morph_class}_avg_circularity'] = np.mean([m['circularity'] for m in morph_macros])
            morphology_summary[f'{morph_class}_avg_complexity'] = np.mean([m['complexity'] for m in morph_macros])
            morphology_summary[f'{morph_class}_avg_distance_to_tcell_um'] = np.mean([m['distance_to_closest_tcell_um'] for m in morph_macros])
        else:
            morphology_summary[f'{morph_class}_count'] = 0
            morphology_summary[f'{morph_class}_avg_area_um2'] = np.nan
            morphology_summary[f'{morph_class}_avg_solidity'] = np.nan
            morphology_summary[f'{morph_class}_avg_circularity'] = np.nan
            morphology_summary[f'{morph_class}_avg_complexity'] = np.nan
            morphology_summary[f'{morph_class}_avg_distance_to_tcell_um'] = np.nan
    
    # Combine all summary stats
    summary_stats.update(morphology_summary)
    
    # Save enhanced summary
    summary_df = pd.DataFrame([summary_stats])
    summary_csv_path = os.path.join(output_dir, "spatial_analysis_summary_detailed.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Enhanced spatial analysis summary saved to: {summary_csv_path}")
    
    # === MORPHOLOGY FEATURE COMPARISON ===
    print("Creating morphology feature comparison...")
    
    # Create a comparison of morphology features across classes
    feature_comparison = []
    
    for morph_class in ['ameboid', 'ramified', 'hyperramified', 'rod']:
        morph_macros = [m for m in macrophage_analysis if m['morphology_classification'] == morph_class]
        if morph_macros:
            features = {
                'morphology_class': morph_class,
                'count': len(morph_macros),
                'avg_area_um2': np.mean([m['area_um2'] for m in morph_macros]),
                'std_area_um2': np.std([m['area_um2'] for m in morph_macros]),
                'avg_perimeter_um': np.mean([m['perimeter_um'] for m in morph_macros]),
                'std_perimeter_um': np.std([m['perimeter_um'] for m in morph_macros]),
                'avg_solidity': np.mean([m['solidity'] for m in morph_macros]),
                'std_solidity': np.std([m['solidity'] for m in morph_macros]),
                'avg_eccentricity': np.mean([m['eccentricity'] for m in morph_macros]),
                'std_eccentricity': np.std([m['eccentricity'] for m in morph_macros]),
                'avg_circularity': np.mean([m['circularity'] for m in morph_macros]),
                'std_circularity': np.std([m['circularity'] for m in morph_macros]),
                'avg_complexity': np.mean([m['complexity'] for m in morph_macros]),
                'std_complexity': np.std([m['complexity'] for m in morph_macros]),
                'avg_aspect_ratio': np.mean([m['aspect_ratio'] for m in morph_macros]),
                'std_aspect_ratio': np.std([m['aspect_ratio'] for m in morph_macros]),
                'avg_roundness': np.mean([m['roundness'] for m in morph_macros]),
                'std_roundness': np.std([m['roundness'] for m in morph_macros]),
                'avg_distance_to_tcell_um': np.mean([m['distance_to_closest_tcell_um'] for m in morph_macros]),
                'std_distance_to_tcell_um': np.std([m['distance_to_closest_tcell_um'] for m in morph_macros])
            }
            feature_comparison.append(features)
    
    # Save morphology comparison
    if feature_comparison:
        comparison_df = pd.DataFrame(feature_comparison)
        comparison_csv_path = os.path.join(output_dir, "morphology_feature_comparison.csv")
        comparison_df.to_csv(comparison_csv_path, index=False)
        print(f"Morphology feature comparison saved to: {comparison_csv_path}")
    
    print("Enhanced spatial analysis CSV generation completed!")
    print(f"Generated files:")
    print(f"  - macrophage_centered_analysis_detailed.csv")
    print(f"  - tcell_centered_analysis_detailed.csv") 
    print(f"  - spatial_analysis_summary_detailed.csv")
    print(f"  - morphology_feature_comparison.csv")
    
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
        'tcell_count': results['stats']['tcell_count'],
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
    
    # T cell metrics
    if results.get('tcell_props') and len(results['tcell_props']) > 0:
        tcell_metrics = {
            'metric': ['area', 'perimeter', 'solidity', 'eccentricity'],
            'min': [],
            'max': [],
            'mean': [],
            'median': [],
            'std': []
        }
        
        for metric in tcell_metrics['metric']:
            # Handle both region props objects and dictionaries
            values = []
            for prop in results['tcell_props']:
                if hasattr(prop, metric):
                    values.append(getattr(prop, metric))
                elif isinstance(prop, dict) and metric in prop:
                    values.append(prop[metric])
            
            if values:
                tcell_metrics['min'].append(np.min(values))
                tcell_metrics['max'].append(np.max(values))
                tcell_metrics['mean'].append(np.mean(values))
                tcell_metrics['median'].append(np.median(values))
                tcell_metrics['std'].append(np.std(values))
            else:
                tcell_metrics['min'].append(0)
                tcell_metrics['max'].append(0)
                tcell_metrics['mean'].append(0)
                tcell_metrics['median'].append(0)
                tcell_metrics['std'].append(0)
        
        tcell_metrics_df = pd.DataFrame(tcell_metrics)
        tcell_metrics_df.to_csv(os.path.join(output_dir, "tcell_metrics_summary.csv"), index=False)
    
    # Include experimental parameters
    params = {
        'parameter': [
            'iba1_min_size', 'iba1_max_size', 
            'dapi_min_size', 'dapi_max_size',
            'tcell_min_size', 'tcell_max_size',
            'artifact_area_threshold', 
            'tile_size', 'overlap'
        ],
        'value': [
            results.get('artifact_filtering', {}).get('iba1_min_size', 'N/A'),
            results.get('artifact_filtering', {}).get('iba1_max_size', 'N/A'),
            results.get('artifact_filtering', {}).get('dapi_min_size', 'N/A'),
            results.get('artifact_filtering', {}).get('dapi_max_size', 'N/A'),
            results.get('artifact_filtering', {}).get('tcell_min_size', 'N/A'),
            results.get('artifact_filtering', {}).get('tcell_max_size', 'N/A'),
            results.get('artifact_filtering', {}).get('area_threshold', 'N/A'),
            'N/A',  # tile_size not stored in results
            'N/A'   # overlap not stored in results
        ]
    }
    
    params_df = pd.DataFrame(params)
    params_df.to_csv(os.path.join(output_dir, "analysis_parameters_summary.csv"), index=False)
    
    return report_df


if __name__ == "__main__":
    # Example usage
    image_path = "/rsrch5/home/plm/yshokrollahi/yasin-vitamin-p/notebooks/sample3_4channel_composite_image.tif"
    output_dir = "/rsrch5/home/plm/yshokrollahi/yasin-vitamin-p/notebooks/sample3_4channel_07072025"
    
    # Run the analysis with 4 channels: QKI (0), IBA1 (1), DAPI (2), T cells (3)
    results = main(
        image_path=image_path,
        output_dir=output_dir,
        qki_channel=3,
        iba1_channel=1,
        dapi_channel=0,
        tcell_channel=2,
        iba1_threshold_method="adaptive",
        iba1_min_size=50,
        iba1_max_size=5000,
        tcell_min_size=50,
        tcell_max_size=250,
        tile_size=2000,
        overlap=200,
        max_workers=16,
        use_enhanced_iba1=True,
        use_enhanced_tcell=True,
        remove_artifacts=True,
        artifact_area_threshold=8000,
        iba1_min_intensity=None,
        iba1_max_intensity=None,
        tcell_min_intensity=None,
        tcell_max_intensity=None
    )
    
    print("Analysis complete!")