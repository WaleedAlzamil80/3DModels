## Evaluation Metrics

### 1. **Chamfer Distance**

The **Chamfer Distance** between two point clouds \( P_1 \) and \( P_2 \) measures the average distance from each point in one point cloud to the closest point in the other point cloud. It is often used in 3D shape matching and reconstruction because it is smooth and differentiable.

#### Mathematically:

Given two point clouds:
- \( P_1 = \{ p_1^i \}_{i=1}^{N} \), where \( p_1^i \in \mathbb{R}^3 \), and \( N \) is the number of points in the first point cloud.
- \( P_2 = \{ p_2^j \}_{j=1}^{M} \), where \( p_2^j \in \mathbb{R}^3 \), and \( M \) is the number of points in the second point cloud.

The **Chamfer Distance** is defined as:
\[
\text{Chamfer}(P_1, P_2) = \frac{1}{N} \sum_{i=1}^{N} \min_{j} \| p_1^i - p_2^j \|_2^2 + \frac{1}{M} \sum_{j=1}^{M} \min_{i} \| p_2^j - p_1^i \|_2^2
\]

#### Explanation:
- The first term \( \frac{1}{N} \sum_{i=1}^{N} \min_{j} \| p_1^i - p_2^j \|_2^2 \) represents the average squared distance from each point in \( P_1 \) to the nearest point in \( P_2 \).
- The second term \( \frac{1}{M} \sum_{j=1}^{M} \min_{i} \| p_2^j - p_1^i \|_2^2 \) does the same, but for points in \( P_2 \) to the nearest point in \( P_1 \).

This ensures symmetry and gives a measure of how close the two point clouds are to each other on average.

#### Properties:
- **Symmetric**: Chamfer Distance is symmetric because it measures the distance from \( P_1 \) to \( P_2 \) and vice versa.
- **Smooth**: Chamfer Distance is differentiable, which makes it ideal for training neural networks via gradient descent.
- **Averages Distances**: It smooths out differences by averaging them, so it is robust to small discrepancies in the point clouds.

---

### 2. **Hausdorff Distance**

The **Hausdorff Distance** between two point clouds measures the **maximum** distance from any point in one point cloud to the nearest point in the other point cloud. Unlike Chamfer Distance, which averages distances, Hausdorff Distance focuses on the worst-case scenario, capturing the largest possible deviation between the point clouds.

#### Mathematically:

Given two point clouds:
- \( P_1 = \{ p_1^i \}_{i=1}^{N} \), where \( p_1^i \in \mathbb{R}^3 \), and \( N \) is the number of points in the first point cloud.
- \( P_2 = \{ p_2^j \}_{j=1}^{M} \), where \( p_2^j \in \mathbb{R}^3 \), and \( M \) is the number of points in the second point cloud.

The **directed Hausdorff Distance** from \( P_1 \) to \( P_2 \) is:
\[
d_H(P_1, P_2) = \max_{i} \min_{j} \| p_1^i - p_2^j \|_2
\]
- \( d_H(P_1, P_2) \) is the largest minimum distance from a point in \( P_1 \) to any point in \( P_2 \).
  
The **symmetric Hausdorff Distance** (which is typically used) is:
\[
\text{Hausdorff}(P_1, P_2) = \max \{ d_H(P_1, P_2), d_H(P_2, P_1) \}
\]
- This takes the maximum of the directed distances between \( P_1 \) and \( P_2 \), and vice versa.

#### Explanation:
- \( \min_{j} \| p_1^i - p_2^j \|_2 \) finds the closest point in \( P_2 \) to a given point \( p_1^i \in P_1 \).
- \( \max_{i} \) takes the maximum of those distances to ensure that the **worst-case distance** is captured.
- Similarly, \( d_H(P_2, P_1) \) captures the worst-case distance from \( P_2 \) to \( P_1 \), and the final **Hausdorff Distance** is the larger of the two.

#### Properties:
- **Sensitive to Outliers**: Hausdorff Distance focuses on the maximum deviation between the point clouds, so it is more sensitive to outliers than Chamfer Distance.
- **Non-Smooth**: The maximum operation makes Hausdorff Distance non-differentiable, which can be problematic for use in optimization tasks like deep learning.
- **Worst-Case Measure**: It is a **worst-case measure** and captures the largest mismatch between the point clouds.

---

### Comparison

| **Property**                | **Chamfer Distance**                         | **Hausdorff Distance**                         |
|-----------------------------|----------------------------------------------|-----------------------------------------------|
| **Definition**               | Measures the average squared distance between points in two point clouds. | Measures the maximum distance between points in two point clouds. |
| **Symmetry**                 | Yes (considers distances from \( P_1 \) to \( P_2 \) and \( P_2 \) to \( P_1 \)). | Yes (max of both directed distances). |
| **Sensitivity to Outliers**  | Robust to outliers because it averages the distances. | Highly sensitive to outliers (focuses on worst-case distance). |
| **Smoothness**               | Smooth and differentiable.                   | Non-differentiable (maximum operation). |
| **Use Case**                 | Suitable for learning tasks (e.g., training deep learning models) due to its smooth nature. | Good for evaluation where maximum deviation matters (e.g., surface matching). |

### Summary:
- **Chamfer Distance** averages distances, making it **smoother** and more **robust to outliers**, which makes it suitable as a loss function for training deep learning models.
- **Hausdorff Distance** captures the **worst-case deviation**, making it sensitive to outliers but useful for evaluating the largest mismatch between two shapes, though it's less commonly used as a loss function due to its non-differentiable nature.