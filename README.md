# Self-supervised cost of transport estimation for multimodal path planning

![Pipeline (1)](https://github.com/user-attachments/assets/402570a8-9d4b-452f-8723-7bf639a38f49)

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
- `export_onxx.ipynb` - Notebook for exporting models to ONNX format.
- `export_pcd_ply.ipynb` - Notebook for exporting point clouds.
- `generate_confidence.py` - Script for generating confidence scores.
- `generate_dataset.py` - Script for dataset generation.
- `generate_valid.ipynb` - Notebook for generating validation data.
- `infer.py` - Script for running inference.
- `inference.py` - Script for inference with additional options.
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

```
run commands in packages.txt
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
3. **Confidence Annotation**: Annotate confidence scores using `annotate_confidence_v2.py`.
```
run the annotate confidence notebook
```

4. **Training**: Train your models using `train.py`.
   ```bash
   python train.py
   ```
5. **Export ONNX**: Run export_onnx to create a onnx model.
   ```
   run export_onxx.ipynb 
   ```
6. **Export trt**: Use trtexec to optimize the model for fast inference
   ```bash
   /usr/src/tensorrt/bin/trtexec --onnx=asymformer-v6.onnx --saveEngine=asymformer-v6.trt
   ```

7. **Infer using the  traversability estimation repo**: 
```
see https://github.com/VinceGHER/traversability_estimation
 ```

### Notebooks

- Use the various Jupyter notebooks (`.ipynb` files) for detailed analysis, exporting models, and testing.

## Contributing

Feel free to fork this repository and contribute by submitting pull requests. Please ensure that your contributions adhere to the project's coding standards.

## Contact

For any questions or suggestions, please open an issue or contact at vgherold@caltech.edu
