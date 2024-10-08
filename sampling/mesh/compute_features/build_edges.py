import trimesh
import numpy as np
import torch

def build_edges(faces):
    """
    Extract edges from the triangular faces.
    Faces are given as an array of vertex indices.
    Returns a dictionary of unique edges and an edge-to-face adjacency list.
    """
    edges = {}  # Stores unique edges (sorted vertex pairs)
    edge_to_faces = {}  # Maps each edge to the faces it belongs to

    for i, face in enumerate(faces):
        # There are 3 edges in each triangle face (i.e., a triangle)
        for j in range(3):
            # Create edges by connecting vertices, ensuring unique ordering
            edge = tuple(sorted([face[j], face[(j+1)%3]]))  # Sorted vertex pair to ensure uniqueness
            if edge not in edges:
                edges[edge] = len(edges)  # Assign an index to the new edge
                edge_to_faces[edge] = [i]  # Track which face this edge belongs to
            else:
                edge_to_faces[edge].append(i)  # This edge also belongs to another face

    return edges, edge_to_faces

def build_edges_torch(faces):
    """
    Efficiently build edges from faces using vectorized operations with PyTorch.
    
    Parameters:
    - faces: A PyTorch tensor of shape (num_faces, 3), where each row represents
             the indices of vertices forming a triangular face.

    Returns:
    - edges: A PyTorch tensor of shape (num_edges, 2), where each row is a sorted pair
             of vertex indices representing an edge.
    """
    # Step 1: Extract edges from the faces tensor
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)

    # Step 2: Sort each edge so that we always have (min, max) order
    edges, _ = torch.sort(edges, dim=1)

    # Step 3: Remove duplicates using torch.unique
    edges = torch.unique(edges, dim=0)

    return edges


def build_edges_vectorized(faces):
    """
    Efficiently build edges from faces using vectorized operations.

    Parameters:
    - faces: A NumPy array of shape (num_faces, 3), where each row represents
             the indices of vertices forming a triangular face.

    Returns:
    - edges: A NumPy array of shape (num_edges, 2), where each row is a sorted pair
             of vertex indices representing an edge.
    - faces_per_edge: A NumPy array of shape (num_edges, 2), where each row contains
                      the indices of the two faces that share the edge. If an edge is
                      on the boundary, the second face will be marked with -1.
    """
    num_faces = faces.shape[0]
    
    # Step 1: Extract all edges (each face has 3 edges)
    edges = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], axis=0)

    # Create a corresponding list of face indices
    face_indices = np.repeat(np.arange(num_faces), 3)

    # Step 2: Sort the vertex indices in each edge so (min, max) order
    edges = np.sort(edges, axis=1)

    # Step 3: Remove duplicate edges by using np.unique, but keep the index mapping
    edges, unique_indices = np.unique(edges, axis=0, return_inverse=True)

    # Step 4: Initialize an array to store faces per edge (2 faces per edge, pad with -1 for boundary edges)
    faces_per_edge = -np.ones((edges.shape[0], 2), dtype=int)

    # Step 5: Fill the faces_per_edge array
    for i, edge_idx in enumerate(unique_indices):
        if faces_per_edge[edge_idx, 0] == -1:
            faces_per_edge[edge_idx, 0] = face_indices[i]
        else:
            faces_per_edge[edge_idx, 1] = face_indices[i]

    return edges, faces_per_edge


def vis(edges, s, threshold, mesh):

    # Separate edges based on the threshold
    above_threshold = edges[s > threshold]
    below_threshold = edges[s <= threshold]
   # Check if there are any edges above or below the threshold
    if len(above_threshold) == 0:
        print("No edges found above the threshold.")
    if len(below_threshold) == 0:
        print("No edges found below the threshold.")

    # Create a list of line segments (each edge as a pair of points)
    edges_lines_above = np.array([[mesh.vertices[e[0]], mesh.vertices[e[1]]] for e in above_threshold])
    edges_lines_below = np.array([[mesh.vertices[e[0]], mesh.vertices[e[1]]] for e in below_threshold])

    # Check if there are any edges to visualize
    if edges_lines_above.shape[0] == 0 or edges_lines_below.shape[0] == 0:
        print("No edges to visualize.")
        return


    # Create edge paths using trimesh.load_path and assign colors directly
    edge_path_above = trimesh.load_path(edges_lines_above)
    edge_path_below = trimesh.load_path(edges_lines_below)

    # edge_path_above.colors = np.array([0, 0, 0, 255] * (len(edges))).reshape(-1, 1)
    # edge_path_below.colors = np.array([255, 0, 255, 0] * (len(edges))).reshape(-1, 1)

    # Create a scene and add both the mesh and the colored edges
    scene = trimesh.Scene([mesh, edge_path_below])

    # Show the scene with both mesh and edges
    scene.show()