# Detailed Methods Explanation

[Image Analysis Methodology
Sample Processing and Imaging
Three tissue samples were analyzed to quantify microglia morphology, T cell distribution, and Qki expression using four-channel fluorescence microscopy (QKI, IBA1, DAPI, CD3). Images were acquired at 0.35 μm/pixel resolution and processed using custom Python-based algorithms implemented with scikit-image [1] OpenCV, and SciPy libraries.
Cell Segmentation
IBA1+ microglia/macrophages were segmented using multi-scale detection combining CLAHE contrast enhancement [2] with dual Gaussian filtering (σ=2 for cell bodies, σ=0.5 for processes). Otsu thresholding was supplemented with adaptive thresholding (block size 31×31 pixels) to capture morphological heterogeneity. Morphological reconstruction with distance transform-based markers reconnected fragmented cells, followed by watershed segmentation [3] to separate touching cells. Size filters (50-5,000 pixels; 6-175 μm²) and shape-based criteria (solidity >0.2, aspect ratio <10) excluded artifacts.
T cells were detected using CLAHE enhancement, rolling-ball background subtraction, and combined global-adaptive thresholding optimized for smaller, circular cells (20-2,000 pixels; 2-70 μm²). DAPI+ nuclei were identified by Otsu thresholding with size filtering (15-500 pixels; 2-17 μm²).
Large images were processed as overlapping tiles (2,000×2,000 pixels, 200-pixel overlap) in parallel with 16 workers, with spatial hashing-based deduplication (10-pixel grid) to eliminate boundary artifacts.
Morphological Classification
IBA1+ cells were classified into four morphological categories based on quantitative shape descriptors. For each cell, we calculated: solidity (ratio of cell area to convex hull area), circularity (4πA/P², where A is area and P is perimeter), complexity (perimeter divided by square root of area), and eccentricity (elongation measure). Classification thresholds were derived from the population distribution using the 25th, 50th, and 75th percentiles of each metric.
Cells were categorized as: ameboid (compact, rounded morphology: high solidity >75th percentile, high circularity >70th percentile, low complexity <25th percentile); rod-shaped (elongated profiles: moderate solidity 50th-75th percentile, high eccentricity >75th percentile, roundness <0.5); ramified (moderately branched processes: low solidity <25th percentile, intermediate complexity 25th-75th percentile); and hyperramified (extensively branched morphology: very high complexity >75th percentile, low solidity, high eccentricity >0.7). When shape metrics yielded ambiguous classifications, cells were assigned using a composite scoring system that summed normalized values of all four descriptors. Segmentation quality was verified through visual inspection of representative images by two independent observers.
Qki Expression and Spatial Analysis
Nuclear Qki intensity was measured within DAPI+ masks inside IBA1+ regions. Cytoplasmic Qki was quantified in a 3-pixel (1.05 μm) annular expansion around each nucleus within the IBA1+ cell boundary. Nuclear-to-cytoplasmic (N:C) ratios were calculated for cells where cytoplasmic signal exceeded background (5th percentile of image intensity).
Spatial relationships were analyzed bidirectionally: each microglia was characterized by its distance to the nearest T cell, and each T cell by the distances and morphological classes of its five nearest microglia. Pairwise Euclidean distances were computed in micrometers. Population-level spatial correlation between cell types was assessed using 2D Pearson correlation of Gaussian-smoothed density maps (σ=30 pixels, 10.5 μm).
All measurements, cell coordinates, and morphological features were exported as structured data for statistical analysis.
References
1.	van der Walt, S. et al. scikit-image: image processing in Python. PeerJ 2, e453 (2014).
2.	Pizer, S. M. et al. Adaptive histogram equalization and its variations. Comput. Vis. Graph. Image Process. 39, 355-368 (1987).
3.	Vincent, L. & Soille, P. Watersheds in digital spaces: an efficient algorithm based on immersion simulations. IEEE Trans. Pattern Anal. Mach. Intell. 13, 583-598 (1991).

]

See the main paper for the concise methods section.
