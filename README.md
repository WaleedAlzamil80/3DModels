# 3D Models Repository

The goal is to create a collection of different 3D models and pipelines that can be applied to similar tasks involving 3D point cloud data, meshes, or volumetric data.

## Introduction
This repository is dedicated to implementing and experimenting with various 3D models. The primary focus is on exploring different architectures for 3D data processing, such as PointNet, and applying them to real-world tasks like 3D teeth segmentation.

This repository is focused on the implementation of various 3D model architectures and methods for processing point clouds and meshes. It currently supports the **PointNet** network, with plans to include other architectures like PointNet++, DGCNN, and MeshCNN in future updates.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Implemented Models](#implemented-models)
3. [Upcoming Features](#upcoming-features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [Repository Structure](#repository-structure)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview

This repository aims to provide implementations for several state-of-the-art neural networks designed to handle 3D data. The focus is on point clouds and meshes, widely used in various domains like computer vision, robotics, and medical imaging.

The repository is structured to allow easy integration of additional methods and architectures. The current implementation includes **PointNet**, a popular neural network model for point cloud data classification and segmentation.

## Implemented Models

- **PointNet**: 
  - **Description**: PointNet is a pioneering architecture for directly processing point clouds. It is used for tasks like classification and segmentation of 3D data.
  - **Files**:
    - `models/PointNet.py`: Contains the implementation of the PointNet architecture.
    - `train.py`: Training script for the PointNet model.
    - `test_PointNet.py`: Testing and evaluation script for PointNet.
    - `losses/PointNetLosses.py`: Custom loss functions for PointNet.

## Upcoming Features

- **Add more 3D models**: Future implementations may include architectures like PointNet++, DGCNN, and 3D CNNs for point cloud and mesh data.
- **Improve segmentation results**: Fine-tune the models to enhance segmentation performance on the teeth dataset and Shapenet.
- **Expand dataset**: Experiment with other 3D datasets and tasks.


We plan to add the following models and architectures in future updates:

- **PointNet++**
- **DGCNN** (Dynamic Graph CNN)
- **MeshCNN**
- **PCT** (Point Cloud Transformer)
- **PointTe**
- **SO-Net**
- **GCN for meshes**
- **3D GAN architectures**

## Installation

To set up the environment and install all the required dependencies, you can use the provided Bash script.

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/3DModels.git
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

### Training the PointNet Model

To train the PointNet model on a dataset, use the `train.py` script:

```bash
python train.py --dataset /path/to/dataset --epochs 50 --batch_size 32
```

You can customize the training configuration by modifying the arguments like dataset path, number of epochs, and batch size.


## Repository Structure

```plaintext
.
├── Dataset/                   # Directory for custom datasets
├── images/                    # Contains visual assets for documentation
├── losses/                    # Custom loss functions
├── models/                    # Neural network architectures
├── utils/                     # Utility scripts for dataset preparation, etc.
├── vis/                       # Visualization scripts (plots, etc.)
├── environment.sh             # Bash script for setting up environment
├── main.py                    # Main entry script (if applicable)
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── test_PointNet.py           # Testing script for PointNet
└── train.py                   # Training script for PointNet
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests if you want to improve the repository or add new features/models.

### Notes:
1. **Model Details**: As you add new models or functionalities, you can include their descriptions and instructions in the relevant sections.
2. **Customizable Sections**: You can modify the "Upcoming Features" and "Usage" sections as you make progress on other models and architectures.
3. **Environment Setup**: If you have a more complex setup or additional environment scripts (e.g., for GPU configurations), include those details in the installation instructions.

Feel free to adjust this template to match the exact specifics of your project!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.