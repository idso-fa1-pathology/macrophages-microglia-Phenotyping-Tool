# Macrophages/Microglia Phenotyping Tool

Automated detection, morphological classification, and Qki expression analysis of IBA1+ macrophages/microglia from multi-channel fluorescence microscopy images.

## Features

- **IBA1+ cell detection**: Automated segmentation of macrophages/microglia
- **Morphological classification**: Ameboid, ramified, hyperramified, and rod-shaped phenotypes
- **Qki expression analysis**: Nuclear and cytoplasmic quantification with N:C ratios
- **Comprehensive outputs**: CSV data files and publication-ready visualizations

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/idso-fa1-pathology/macrophages-microglia-phenotyping-tool.git
cd macrophages-microglia-phenotyping-tool
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

# Update channel indices if needed (default is DAPI=0, QKI=1, IBA1=2)
results = main(
    image_path=image_path,
    output_dir=output_dir,
    dapi_channel=0,     # Update if your channels are different
    qki_channel=1,
    iba1_channel=2,
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
    image_path="path/to/your/3channel_image.tif",
    output_dir="output/sample1",
    dapi_channel=0,     # Nuclei (DAPI)
    qki_channel=1,      # QKI protein
    iba1_channel=2      # Macrophages/microglia marker (IBA1)
)

print(f"Found {results['stats']['iba1_count']} IBA1+ cells")
```

## Input Requirements

3-channel fluorescence microscopy images (TIFF format):
- **Channel 0**: DAPI (nuclei staining)
- **Channel 1**: QKI (protein marker)
- **Channel 2**: IBA1 (macrophages/microglia marker)

Recommended resolution: 0.35 μm/pixel

## Output Files

The pipeline generates comprehensive results in the specified output directory:

### CSV Data Files
- `stats.csv` - Overall statistics summary
- `all_cells_comprehensive.csv` - All cells with morphology and expression data
- `iba1_cells.csv` - IBA1+ cell properties
- `morphology_classification.csv` - Morphology distribution summary
- `qki_expression_analysis.csv` - Qki nuclear and cytoplasmic measurements with N:C ratios

### Visualizations
- `composite.png` - Multi-channel overlay with cell boundaries
- `morphology.png` - Color-coded morphology classification
- `iba1_labeled.png` - Segmented IBA1+ cells
- `morphology_pie_chart.png` - Morphology distribution chart
- `qki_expression_heatmap.png` - Qki intensity visualization

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
iba1_min_size=50         # Minimum microglia/macrophage size (6 μm² at 0.35 μm/pixel)
iba1_max_size=5000       # Maximum microglia/macrophage size (175 μm² at 0.35 μm/pixel)

# Processing options
tile_size=2000           # Tile size for parallel processing
overlap=200              # Overlap between tiles
max_workers=16           # Number of parallel workers

# Detection options
use_enhanced_iba1=True   # Use advanced detection algorithm
remove_artifacts=True    # Apply artifact filtering (solidity >0.2, aspect ratio <10)
```

## Morphology Classification

IBA1+ macrophages/microglia are classified into four phenotypes based on quantitative shape descriptors:

- **Ameboid**: Compact, rounded morphology (high solidity >75th percentile, high circularity >70th percentile, low complexity <25th percentile) - activated/inflammatory state
- **Rod-shaped**: Elongated profiles (moderate solidity 50th-75th percentile, high eccentricity >75th percentile, roundness <0.5) - reactive/injury-associated state
- **Ramified**: Moderately branched processes (low solidity <25th percentile, intermediate complexity 25th-75th percentile) - surveillance state
- **Hyperramified**: Extensively branched morphology (very high complexity >75th percentile, low solidity, high eccentricity >0.7) - highly surveillant state

Classification uses population-based percentile thresholds (25th, 50th, 75th) derived from solidity, circularity, complexity, and eccentricity metrics. When shape metrics yield ambiguous classifications, cells are assigned using a composite scoring system that sums normalized values of all four descriptors.

## Qki Expression Analysis

Nuclear Qki intensity is measured within DAPI+ masks inside IBA1+ regions. Cytoplasmic Qki is quantified in a 3-pixel (1.05 μm at 0.35 μm/pixel resolution) annular expansion around each nucleus within the IBA1+ cell boundary. Nuclear-to-cytoplasmic (N:C) ratios are calculated for cells where cytoplasmic signal exceeds background (5th percentile of image intensity).

## Citation

If you use this tool in your research, please cite:
```
[Citation will be added upon publication]
```

## Methodology

### Cell Segmentation
IBA1+ macrophages/microglia are segmented using multi-scale detection combining CLAHE contrast enhancement [2] with dual Gaussian filtering (σ=2 for cell bodies, σ=0.5 for processes). Otsu thresholding is supplemented with adaptive thresholding (block size 31×31 pixels) to capture morphological heterogeneity. Morphological reconstruction with distance transform-based markers reconnects fragmented cells, followed by watershed segmentation [3] to separate touching cells. Size filters (50-5,000 pixels; 6-175 μm²) and shape-based criteria (solidity >0.2, aspect ratio <10) exclude artifacts.

Large images are processed as overlapping tiles (2,000×2,000 pixels, 200-pixel overlap) in parallel with 16 workers, with spatial hashing-based deduplication (10-pixel grid) to eliminate boundary artifacts.

Detailed methodology is described in the manuscript methods section.

### References
1. van der Walt, S. et al. scikit-image: image processing in Python. PeerJ 2, e453 (2014).
2. Pizer, S. M. et al. Adaptive histogram equalization and its variations. Comput. Vis. Graph. Image Process. 39, 355-368 (1987).
3. Vincent, L. & Soille, P. Watersheds in digital spaces: an efficient algorithm based on immersion simulations. IEEE Trans. Pattern Anal. Mach. Intell. 13, 583-598 (1991).

## Requirements

- Python 3.8+
- 16GB+ RAM recommended for large images
- Multi-core CPU for parallel processing

## Troubleshooting

**Memory errors with large images:**
- Reduce `tile_size` parameter (e.g., 1500 or 1000)
- Reduce `max_workers` parameter

**Too many/few cells detected:**
- Adjust size filters (`iba1_min_size`, `iba1_max_size`)
- Modify intensity thresholds (`iba1_min_intensity`, `iba1_max_intensity`)

**Poor segmentation quality:**
- Check channel indices match your image
- Verify image quality and contrast
- Adjust detection parameters in code

**"Channel out of bounds" error:**
- Your image may have fewer than 3 channels
- Check actual channel count with: `python check_image.py`
- Adjust channel parameters accordingly

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub or contact the development team.

## Acknowledgments

This tool was developed for automated analysis of macrophages/microglia morphology and Qki expression in brain tissue imaging studies.
