# Models

This repository implements various deep learning models for processing 3D point clouds, focusing on tasks like classification and segmentation. Below is a summary of the models implemented and planned for this repository.

---

### 1. PointNet
PointNet is a deep learning architecture designed for directly processing 3D point clouds. It uses MLPs and symmetric functions to capture both local and global features from 3D data. This architecture is well-suited for tasks like 3D classification and segmentation.

- **Paper**: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- **Current Implementation**: The current version implements PointNet for 3D teeth segmentation, focusing on direct feature extraction from point clouds.

![PointNet](/images/PointNet/PointNet.png)

---

### 2. PointNet++
PointNet++ builds upon the foundation of PointNet by introducing hierarchical feature learning to capture both local and global features at multiple scales. It uses a combination of set abstraction layers and sampling techniques to better represent fine details in 3D point clouds.

- **Paper**: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
- **Current Implementation**: The PointNet++ implementation has been successfully integrated for more robust 3D segmentation and classification tasks, with enhanced local feature extraction capabilities, making it more suited for complex datasets like 3D teeth.
- **Future Enhancements**:
  - Multi-scale grouping (MSG)
  - Multi-resolution grouping (MRG)

![PointNet++](/images/PointNet/PointNetpp.png)

---

### 3. DynamicGraphCNN
DynamicGraphCNN (DGCNN) is a graph-based neural network for 3D point cloud processing. It dynamically constructs a graph in feature space for each layer and captures geometric relationships between points.

- **Paper**: [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829)
- **Current Implementation**: DGCNN is implemented for segmentation tasks, leveraging dynamic graph construction and edge convolutions to capture rich geometric information.
- **Future Enhancements**:
  - Optimization for large-scale datasets
  - Improved memory efficiency

![DynamicGraphCNN](/images/Graphs/DynamicGraphCNN.png)

---

### 4. Point Cloud Transformer (PCT)
The Point Cloud Transformer adapts the transformer architecture for 3D point clouds, enabling global context learning. It replaces traditional MLP layers with attention mechanisms to capture complex point relationships.

- **Paper**: [PCT: Point Cloud Transformer](https://arxiv.org/abs/2012.09688)
- **Current Implementation**: PCT is successfully integrated, focusing on segmentation tasks with attention-based feature extraction.

![PCT](/images/transformers/PCT/PCTArchitecture.png)
---

### 5. Mining Point Cloud
This model explores techniques for data mining and feature extraction in point clouds to optimize segmentation and classification tasks. It includes methods for data augmentation, denoising, and efficient sampling.

- **Paper**: [Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling](https://arxiv.org/pdf/1712.06760)
- **Current Implementation**: FoldingNet is implemented as a standalone feature extractor and has been used with the Spatial Transformer introduced in the PointNet paper.

![Mining Point Cloud](/images/Graphs/MiningPointCloud.png)

---

## Usage  
This repository provides a unified interface for all implemented models. You can easily switch between models and tasks using the factory pattern.

Import the desired model in your script:  
```python  
from factories.model_factory import get_model

# Example: Load DynamicGraphCNN for segmentation task with 33 output classes
model = get_model("DynamicGraphCNN", "segmentation", 33)
