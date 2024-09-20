## Implemented Models

### 1. PointNet
PointNet is a deep learning architecture designed for directly processing 3D point clouds. It uses MLPs and symmetric functions to capture both local and global features from 3D data. This architecture is well-suited for tasks like 3D classification and segmentation.

- **Paper**: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- **Current Implementation**: The current version implements PointNet for 3D teeth segmentation, focusing on direct feature extraction from point clouds.

### 2. PointNet++
PointNet++ builds upon the foundation of PointNet by introducing hierarchical feature learning to capture both local and global features at multiple scales. It uses a combination of set abstraction layers and sampling techniques to better represent fine details in 3D point clouds.

- **Paper**: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
- **Current Implementation**: The PointNet++ implementation has been successfully integrated for more robust 3D segmentation and classification tasks, with enhanced local feature extraction capabilities, making it more suited for complex datasets like 3D teeth.