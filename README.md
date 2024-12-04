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
- **`infer_segmentation.py`**: The main script to infere a trained model on a file.  
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

To train a model on your dataset, use the `main.py` script with the following command:

```bash
python3 main.py \
    --path "/path_to_dataset_folder" \
    --test_ids "/home/waleed/Documents/3DLearning/3DModels/test_final_data.txt" \
    --n_centroids 1024 \
    --knn 16 \
    --clean \
    --nsamples 1 \
    --batch_size 2 \
    --num_workers 4 \
    --sampling "fpsample" \
    --p 1 \
    --num_epochs 5 \
    --model "PCT" \
    --loss "crossentropy" \
    --rigid_augmentation_train \
    --rotat 1 \
    --k 33
```

### Explanation of Key Arguments:
- `--path`: Path to the dataset directory.
- `--test_ids`: File containing test dataset IDs.
- `--k`: Number of classes for segmentation or classification.
- `--n_centroids`: Number of centroids for sampling.
- `--nsamples`: Number of nearest neighbors or sample points.
- `--knn`: Number of nearest neighbors for dynamic graph construction.
- `--clean`: Flag to clean the dataset by removing unnecessary points.
- `--sampling`: Sampling technique (e.g., `fps`, `voxilization`).
- `--batch_size`: Number of samples per batch during training.
- `--rotat`: Degree of rotation (rotated randomlly before feeding to the model withen range `[-rotat*pi, rotat*pi]`).
- `--num_epochs`: Number of epochs for training.
- `--model`: The model architecture to use (e.g., `DynamicGraphCNN`, `PCT`).
- `--loss`: Loss function to optimize (e.g., `crossentropy`, `focal`,  `dice`).

You can customize these arguments to suit your training configuration.

---

### Inference

To perform inference using a trained model, run the `infer_segmentation.py` script:

```bash
python3 infer_segmentation.py \
    --model "PCT" \
    --pretrained "/path_to_checkpoint/model_checkpoint.pth" \
    --path "/path_to_input_file.bmesh" \
    --clean \
    --p 0 \
    --sample \
    --sampling "fpsample" \
    --n_centroids 1024 \
    --nsamples 16 \
    --visualize \
    --test \
    --test_ids "/path_to_test_file.json" \
    --k 33

```

### Explanation of Key Arguments:
- `--model`: Model architecture used for inference.
- `--pretrained`: Path to the pretrained model checkpoint.
- `--path`: Path to the input file (e.g., `.bmesh` format).
- `--k`: Number of classes for segmentation or classification.
- `--clean`: Flag to clean data points during preprocessing.
- `--p`: Indicate if the file is lower jaw `0` or upper jaw `1`.
- `--sample`: Flag to enable sampling from the entire input file.
- `--sampling`: Sampling technique used (e.g., `fpsample`).
- `--n_centroids`: Number of centroids used for sampling.
- `--nsamples`: Number of nearest neighbors or sample points.
- `--knn`: Number of nearest neighbors for dynamic graph construction.
- `--visualize`: Flag to visualize the segmentation results.
- `--test`: Flag to compare results against ground truth.
- `--test_ids`: Path to the ground truth of the file (e.g., `.json` format).

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
