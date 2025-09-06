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

2. Create a new virtual environment using Python<3.12 (optional but recommended):
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


Alternatively, you can use `conda` to create an environment:
```
conda env create -f environment.yml
conda activate ai4g-flood
```


## Usage

### Planetary Computer Inference

To run inference using images from the Planetary Computer:

```bash
python src/run_flood_detection_planetary_computer.py \
    --region "your_region" \  # should be a country from ne_110m_admin_0_countries.shp, see src/data/country_boundaries
    --start_date "2023-01-01" \
    --end_date "2023-12-31" \
    --model_path "models/ai4g_sar_model.ckpt" \
    --output_dir "path/to/output/directory" \
    --batch_size 1 \
    --input_size 128 \
    --device_index 0
```

### Local Image Inference

For local image inference, use the `run_flood_detection_downloaded_images.py` script (implementation details to be added).

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
|   ├── run_flood_detection_downloaded_images.py
│   └── run_flood_detection_planetary_computer.py
│
├── models/
│   └── ai4g_sar_model.ckpt
│
├── requirements.txt
└── README.md
```

## Model Details

The flood detection model is based on a U-Net architecture with a MobileNetV2 encoder. It takes SAR imagery (VV and VH polarizations) as input and produces binary flood maps.

## Data

The model uses Sentinel-1 SAR data accessed through the Microsoft Planetary Computer. The `FloodDataModule` class handles data retrieval and preprocessing.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## Citation

If you use find this work useful, please cite our [Nature Communications paper](https://arxiv.org/abs/2411.01411), or on [arxiv](https://arxiv.org/abs/2411.01411).

Bibtex:
```
@article{misra2025mapping,
  title={Mapping global floods with 10 years of satellite radar data},
  author={Misra, Amit and White, Kevin and Nsutezo, Simone Fobi and Straka III, William and Lavista, Juan},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={5762},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
