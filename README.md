# Microglia Phenotyping Tool

Automated detection, morphological classification, and spatial analysis of IBA1+ microglia and T cells from multi-channel fluorescence microscopy images.

## Features

- **Multi-cell detection**: IBA1+ microglia, CD3+ T cells, and DAPI+ nuclei
- **Morphological classification**: Ameboid, ramified, hyperramified, and rod-shaped microglia
- **Qki expression analysis**: Nuclear and cytoplasmic quantification with N:C ratios
- **Spatial analysis**: Distance calculations and co-localization between cell types
- **Comprehensive outputs**: CSV data files and publication-ready visualizations

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/microglia-phenotyping-tool.git
cd microglia-phenotyping-tool
```

### 2. Install dependencies

**Using pip:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Using conda:**
```bash
conda env create -f environment.yml
conda activate microglia-analysis
```

### 3. Test the installation


Open `test_analysis.py` and update the image path and channel configuration:
```python
# Update these lines in test_analysis.py:
image_path = "path/to/your/image.tif"  # Change this to your image
output_dir = "results/my_sample"        # Change output directory name

# Update channel indices if needed (default is DAPI=0, QKI=1, IBA1=2, T-cell=3)
results = main(
    image_path=image_path,
    output_dir=output_dir,
    dapi_channel=0,     # Update if your channels are different
    qki_channel=1,
    iba1_channel=2,
    tcell_channel=3,
    max_workers=4       # Adjust based on your CPU cores
)
```

Run the test:
```bash
python test_analysis.py
```

If successful, you should see:
```
✅ ALL TESTS PASSED!
   README instructions work correctly
   Results saved in: results/my_sample/
```

## Quick Start

After installation, you can run analysis on your images:
```python
from cell_analysis import main

results = main(
    image_path="path/to/your/4channel_image.tif",
    output_dir="output/sample1",
    qki_channel=1,      # QKI protein
    iba1_channel=2,     # Microglia marker (IBA1)
    dapi_channel=0,     # Nuclei (DAPI)
    tcell_channel=3     # T cell marker (CD3)
)

print(f"Found {results['stats']['iba1_count']} microglia")
print(f"Found {results['stats']['tcell_count']} T cells")
```

## Input Requirements

4-channel fluorescence microscopy images (TIFF format):
- **Channel 0**: DAPI (nuclei staining)
- **Channel 1**: QKI (protein marker)
- **Channel 2**: IBA1 (microglia/macrophage marker)
- **Channel 3**: CD3 (T cell marker)

Recommended resolution: 0.35 μm/pixel

## Output Files

The pipeline generates comprehensive results in the specified output directory:

### CSV Data Files
- `stats.csv` - Overall statistics summary
- `all_cells_comprehensive.csv` - All cells with morphology and expression data
- `iba1_cells.csv` - IBA1+ cell properties
- `tcells.csv` - T cell properties
- `morphology_classification.csv` - Morphology distribution summary
- `macrophage_centered_analysis_detailed.csv` - Microglia-centric spatial analysis
- `tcell_centered_analysis_detailed.csv` - T cell-centric spatial analysis
- `spatial_analysis_summary_detailed.csv` - Population-level spatial statistics

### Visualizations
- `composite.png` - Multi-channel overlay with cell boundaries
- `morphology.png` - Color-coded morphology classification
- `iba1_labeled.png` - Segmented IBA1+ cells
- `tcell_labeled.png` - Segmented T cells
- `combined_density_heatmap.png` - Spatial density visualization
- `morphology_pie_chart.png` - Morphology distribution chart
- `cell_count_comparison.png` - Cell count bar chart

## Usage Examples

See `example_notebook.ipynb` for detailed tutorials including:
- Basic analysis workflow
- Parameter customization
- Batch processing multiple images
- Visualization of results

## Parameters

Key parameters for customization:
```python
# Cell size filters (in pixels)
iba1_min_size=50         # Minimum microglia size
iba1_max_size=5000       # Maximum microglia size
tcell_min_size=20        # Minimum T cell size
tcell_max_size=2000      # Maximum T cell size

# Processing options
tile_size=2000           # Tile size for parallel processing
overlap=200              # Overlap between tiles
max_workers=16           # Number of parallel workers

# Detection options
use_enhanced_iba1=True   # Use advanced detection algorithm
use_enhanced_tcell=True  # Use advanced T cell detection
remove_artifacts=True    # Apply artifact filtering
```

## Morphology Classification

Microglia are classified into four phenotypes based on quantitative shape descriptors:

- **Ameboid**: Compact, rounded morphology (activated/inflammatory state)
- **Rod-shaped**: Elongated profiles (reactive/injury-associated state)
- **Ramified**: Moderately branched processes (surveillance state)
- **Hyperramified**: Extensively branched morphology (highly surveillant state)

Classification uses solidity, circularity, complexity, and eccentricity metrics with population-based percentile thresholds.

## Citation

If you use this tool in your research, please cite:
```
[Citation will be added upon publication]
```

## Methodology

Detailed methodology is described in `docs/methods_explanation.md`. The pipeline uses established computer vision algorithms:
- CLAHE contrast enhancement [Pizer et al. 1987]
- Watershed segmentation [Vincent & Soille 1991]
- scikit-image library [van der Walt et al. 2014]

## Requirements

- Python 3.8+
- 16GB+ RAM recommended for large images
- Multi-core CPU for parallel processing

## Troubleshooting

**Memory errors with large images:**
- Reduce `tile_size` parameter (e.g., 1500 or 1000)
- Reduce `max_workers` parameter

**Too many/few cells detected:**
- Adjust size filters (`iba1_min_size`, `iba1_max_size`, etc.)
- Modify intensity thresholds (`iba1_min_intensity`, `iba1_max_intensity`)

**Poor segmentation quality:**
- Check channel indices match your image
- Verify image quality and contrast
- Adjust detection parameters in code

**"Channel out of bounds" error:**
- Your image may have fewer than 4 channels
- Check actual channel count with: `python check_image.py`
- Adjust channel parameters accordingly

## License

MIT License - See LICENSE file for details

## Contact


## Acknowledgments

This tool was developed for automated analysis of microglia and T cell interactions in brain tissue imaging studies.
