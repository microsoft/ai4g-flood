# Flood Detection Model

This repository contains a flood detection model that uses Synthetic Aperture Radar (SAR) imagery from the Planetary Computer to identify flooded areas. The model processes Sentinel-1 data and can be used for both local image pairs and large-scale analysis using Planetary Computer data.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Planetary Computer Inference](#planetary-computer-inference)
  - [Local Image Inference](#local-image-inference)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Data](#data)
- [Contributing](#contributing)
- [Citation](#citation)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/microsoft/ai4g-flood.git
   cd ai4g-flood
   ```

2. Create a new virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install the package in editable mode:
   ```
   pip install -e .
   ```

## Usage

### Planetary Computer Inference

To run inference using images from the Planetary Computer:

```bash
python src/run_flood_detection.py \
    --region "your_region" \
    --start_date "2023-01-01" \
    --end_date "2023-12-31" \
    --model_path "path/to/your/model.pth" \
    --output_dir "path/to/output/directory" \
    --batch_size 1 \
    --patch_size 1024 \
    --input_size 128 \
    --mask_zeros
```

### Local Image Inference

For local image inference, use the `local_image_inference.py` script (implementation details to be added).

## Project Structure

```
flood-detection-model/
│
├── src/
│   ├── utils/
│   │   ├── flood_dataset.py
│   │   ├── flood_data_module.py
│   │   ├── image_processing.py
│   │   └── model.py
│   │
│   └── run_flood_detection.py
│
├── models/
│   └── flood_detection_model.pth
│
├── requirements.txt
├── setup.py
└── README.md
```

## Model Details

The flood detection model is based on a U-Net architecture with a MobileNetV2 encoder. It takes SAR imagery (VV and VH polarizations) as input and produces binary flood maps.

## Data

The model uses Sentinel-1 SAR data accessed through the Microsoft Planetary Computer. The `FloodDataModule` class handles data retrieval and preprocessing.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## Citation

If you use find this work useful, please cite our paper: https://arxiv.org/abs/2411.01411
