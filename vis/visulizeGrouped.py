import numpy as np
import matplotlib.pyplot as plt
import trimesh

def visualize_with_trimesh(vertices, labels):
    # Create a color map: each unique label will get a different color
    unique_labels = np.unique(labels)
    colors = plt.get_cmap('hsv', len(unique_labels))(range(len(unique_labels)))

    # Initialize the colors array for the vertices
    vertex_colors = np.zeros((vertices.shape[0], 4))  # RGBA colors

    # Assign colors to vertices based on their labels
    for i, label in enumerate(unique_labels):
        vertex_colors[labels == label] = colors[i]

    # Create a trimesh object for the point cloud
    cloud = trimesh.points.PointCloud(vertices, colors=vertex_colors)

    # Show the point cloud
    cloud.show()