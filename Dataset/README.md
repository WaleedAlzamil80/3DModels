# Dataset

Utilities and scripts for loading, preprocessing, and augmenting datasets.

## Used datasets
The dataset used for initial tests is from the [OSF 3D Teeth Segmentation](https://osf.io/) project. This dataset contains 3D models of teeth, annotated for segmentation tasks.

To download the dataset:
1. Visit the [OSF dataset page](https://osf.io/) (link to the dataset you are using).
2. Download the dataset and place it in the `data/` folder in the repository.

## Supported Formats  
- `.obj`, `.stl`, `.ply`, `.bmesh`: 3D object files.
- `.json`: Label files.

## Functions
- `load_dataset`: Loads the training and testing datasets.  
- `augment_data`: Applies transformations to the data.
