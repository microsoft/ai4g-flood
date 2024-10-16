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
- [License](#license)

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

## License

[Add your chosen license here]

```

This README provides a basic structure and information about your flood detection model project. You may want to expand on certain sections, such as:

1. Adding more detailed usage examples
2. Providing information about the model's performance and any benchmarks
3. Explaining the data preprocessing steps in more detail
4. Adding a section on how to train or fine-tune the model
5. Including information about the project's history or future plans
6. Adding contact information or links to related resources

Feel free to modify and expand this README to best suit your project's needs and to provide the most useful information to potential users of your flood detection model.