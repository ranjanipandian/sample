# Stage-1 Classical Image QC Engine for Microscopy Images

## Overview
A deterministic, modality-agnostic image quality control engine for 2D/3D/4D microscopy TIFF images.
Uses only classical image processing techniques (no AI/ML).

## Features
- **Comprehensive QC Metrics**: 14 categories covering signal quality, blur/focus, noise, artifacts, and more
- **Multi-Dimensional Support**: Handles 2D, 3D (Z-stack), and 4D (time-lapse) images
- **Memory Efficient**: Streaming processing for large datasets
- **Structured Output**: Machine-readable JSON + human-readable reports
- **Optional Visualizations**: Heatmaps, histograms, and Z-profile plots

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python src/main.py --input "D:\Users\2449351\Downloads\Sample_Images\Image1\data_000.tif"
```

### Advanced Options
```bash
python src/main.py \
    --input "path/to/image.tif" \
    --output-dir "./output" \
    --generate-visualizations \
    --verbose
```

## Output Structure
```
output/
├── qc_results/          # JSON files with detailed metrics
└── reports/             # Human-readable text reports
```

## QC Status
- **PASS**: All metrics within acceptable ranges
- **REVIEW**: Some warnings present
- **FAIL**: Critical issues detected

## No AI/ML Policy
This engine uses only classical, deterministic image processing methods from:
- NumPy, SciPy, scikit-image, OpenCV

## License
MIT
