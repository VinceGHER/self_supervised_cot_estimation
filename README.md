# Self-supervised cost of transport estimation for multimodal path planning

## Overview

This paper addresses the challenge of autonomous navigation for robots in real environments by developing a self-supervised learning method that enables a robot to estimate the cost of transport using only vision inputs. The method is applied to the multi-modal mobility morphobot (M4), which can drive, fly, segway, and crawl. By using this approach, the robot can autonomously choose the most energetically efficient mode of locomotion for different environments. The system, which is tested in real-world scenarios, accurately assesses transport costs for various terrains and operates efficiently on an Nvidia Jetson Orin Nano compute unit. This work aims to enhance the navigational and exploratory capabilities of multi-modal robotic platforms.

## Project Structure

- `.gitignore` - Specifies files to be ignored by Git.
- `.gitmodules` - Contains information about submodules.
- `.vscode` - Directory for Visual Studio Code specific settings.
- `__init__.py` - Initializes the module.
- `add_projected.ipynb` - Notebook for adding projections.
- `analyze_dataset.ipynb` - Notebook for dataset analysis.
- `annoate_confidence.ipynb` - Notebook for confidence annotation.
- `annoate_confidence_v2.ipynb` - Updated notebook for confidence annotation.
- `copy_of_crate.obj` - 3D object file.
- `export_onxx.ipynb` - Notebook for exporting models to ONNX format.
- `export_pcd_ply.ipynb` - Notebook for exporting point clouds.
- `generate_confidence.py` - Script for generating confidence scores.
- `generate_dataset.py` - Script for dataset generation.
- `generate_valid.ipynb` - Notebook for generating validation data.
- `infer.py` - Script for running inference.
- `inference.py` - Script for inference with additional options.
- `my_texture_name.png` - Texture image used in the project.
- `old` - Directory containing old versions of scripts and notebooks.
- `packages.txt` - List of required packages.
- `src` - Source code directory.
- `sweep.py` - Script for hyperparameter sweeping.
- `test.ipynb` - Notebook for testing.
- `test.py` - Script for running tests.
- `test_results.ipynb` - Notebook for analyzing test results.
- `testautoencoder.py` - Script for testing autoencoder.

## Getting Started

### Prerequisites

Ensure you have the necessary packages installed. You can install them using:

```bash
pip install -r packages.txt
```

### Running the Project

1. **Data Generation**: Use `generate_dataset.py` to create the dataset.
   ```bash
   python generate_dataset.py
   ```

2. **Confidence Annotation**: Annotate confidence scores using `generate_confidence.py`.
   ```bash
   python generate_confidence.py
   ```

3. **Training**: Train your models using `train.py`.
   ```bash
   python train.py
   ```

4. **Inference**: Run inference using `infer.py` or `inference.py`.
   ```bash
   python infer.py
   ```

### Notebooks

- Use the various Jupyter notebooks (`.ipynb` files) for detailed analysis, exporting models, and testing.

## Contributing

Feel free to fork this repository and contribute by submitting pull requests. Please ensure that your contributions adhere to the project's coding standards.

## License

This project is licensed under the License at the root of the project (See `LICENSE`).

## Contact

For any questions or suggestions, please open an issue or contact at vincent@gherold.com
