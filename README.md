# 3D Models for Segmentation and Generation  

## Overview  
This repository aims to provide implementations for several state-of-the-art neural networks designed to handle 3D data. The focus is on point clouds and meshes, widely used in various domains like computer vision, robotics, and medical imaging.

The primary focus is currently on **segmentation**, implementing 3D models tailored for **teeth segmentation** and **crown generation** tasks, with generation tasks to be addressed in future development phases.

## Features  
- Implementation of 3D segmentation models for dental applications.  
- Modular design for model creation, training, and evaluation.  
- Tools for dataset preprocessing, sampling, and visualization.  

## Directory Structure  

- **`main.py`**: The main script to train or test models.  
- **`requirements.txt`**: Lists all Python dependencies.  
- **`Dataset/`**: Code and utilities for loading and preprocessing datasets.  
- **`models/`**: Contains model architectures for segmentation and future generation tasks.  
- **`train/`**: Training loops and scripts.  
- **`losses/`**: Custom loss functions for segmentation tasks.  
- **`metrics/`**: Evaluation metrics for segmentation performance.  
- **`utils/`**: Utility functions for logging, debugging, etc.  
- **`config/`**: Configuration files for setting up experiments.  
- **`vis/`**: Visualization tools for outputs and intermediate results.  
- **`sampling/`**: Functions for augmenting and sampling datasets.  
- **`images/`**: Contains visualizations or example outputs.  

## Installation

To set up the environment and install all the required dependencies, you can use the provided Bash script.

### Step 1: Clone the Repository

```bash
git clone https://github.com/WaleedALzamil80/3DModels.git
cd 3DModels
```

### Step 2: Create a Virtual Environment

You can use the provided `environment.sh` script to set up your environment:

```bash
bash environment.sh
```

Alternatively, you can manually create a virtual environment and install dependencies:

```bash
python3 -m venv env
source env/bin/activate  # On Windows use .\env\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Install Additional Dependencies (if needed)

```bash
pip install torch torchvision numpy scikit-learn trimesh
```

## Usage

### Training

To train the PointNet model on a dataset, use the `train.py` script:

```bash
python train.py --dataset /path/to/dataset --epochs 50 --batch_size 32
```

You can customize the training configuration by modifying the arguments like dataset path, number of epochs, and batch size.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests if you want to improve the repository or add new features/models.

### Notes:
1. **Model Details**: As you add new models or functionalities, you can include their descriptions and instructions in the relevant sections.
2. **Customizable Sections**: You can modify the "Upcoming Features" and "Usage" sections as you make progress on other models and architectures.
3. **Environment Setup**: If you have a more complex setup or additional environment scripts (e.g., for GPU configurations), include those details in the installation instructions.

Feel free to adjust this template to match the exact specifics of your project!

## Future Work  
- Crown generation models and tools.  
- Enhanced visualization and post-processing methods.  
- Integration with other dental datasets for generalizability.  
- Future implementations may include more advanced architectures.
- Experiment with other 3D datasets and tasks.
- Fine-tune the models to enhance segmentation performance on the teeth dataset and other datasets.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
