## Sampling Techniques for Point Cloud:

| **Algorithm**               | **Time Complexity**         | **Pros**                                                   | **Cons**                                             |
|-----------------------------|-----------------------------|------------------------------------------------------------|------------------------------------------------------|
| **Poisson Disk Sampling**    | **O(n \log n)**             | - Uniform distribution.<br>- Preserves geometry well.       | - Computationally expensive.<br>- Slower. |
| **Voxel Grid Downsampling**  | **O(n)**                    | - Efficient.<br>- Preserves shape well.                     | - Fine details can be lost.<br>- Voxel size selection is crucial. |
| **K-means Clustering**       | **O(n \cdot k \cdot t)**    | - Good spatial distribution.<br>- Global structure preserved. | - Expensive.<br>- Sensitive to initialization. |
| **Curvature-based Sampling** | **O(n \log n)**             | - Focuses on high-detail areas.<br>- Preserves geometry.    | - Requires curvature calculation.<br>- Computationally expensive. |
| **Weighted Random Sampling** | **O(n)** (precomputed weights) | - Flexible and can prioritize important points.<br>- Simple with precomputed weights. | - Requires a good weighting function.<br>- May still miss fine details. |
| **Geodesic Distance Sampling**| **O(n^2)**                 | - Captures global structure.<br>- Ensures even surface coverage.<br>- Preserves geometry. | - Very expensive.<br>- Requires precomputed geodesic distances. |

## Sampling Techniques for Meshes (Faces):

| **Algorithm**                         | **Time Complexity**   | **Pros**                                              | **Cons**                                            | **Best For**                       |
|---------------------------------------|-----------------------|-------------------------------------------------------|-----------------------------------------------------|------------------------------------|
| **Area-Weighted Face Sampling**       | **O(f)**              | Preserves large, important areas.                     | Fine details in smaller faces may be missed.         | Geometry-preserving simplification.|
| **Edge Collapsing (Basic)**           | **O(f \log f)**       | Reduces complexity while preserving structure.        | May distort small, detailed areas.                   | Mesh simplification.               |
| **Quadric Edge Collapse Decimation**  | **O(f \log f)**       | Preserves geometry while reducing face count.         | More expensive than basic edge collapsing.           | High-quality mesh reduction.       |
| **Laplacian-based Mesh Smoothing**    | **O(v)**              | Smooths noise while preserving overall shape.         | Can lead to oversmoothing and loss of fine details.  | Mesh smoothing, not simplification.|
| **Vertex Clustering**                 | **O(v)**              | Fast, reduces both vertices and faces.                | Loses fine details and precision.                    | Coarse mesh simplification.        |
| **Geodesic Distance Sampling**        | **O(f^2)**            | Good global structure preservation.                   | Computationally expensive.                           | Uniform surface remeshing.         |
| **Importance Sampling**               | **O(f)** (after preprocessing)| Focuses on key features like curvature or area.       | Requires feature preprocessing.                      | Feature-preserving simplification. |
