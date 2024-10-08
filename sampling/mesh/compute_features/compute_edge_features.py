import numpy as np
from build_edges import build_edges_vectorized

def compute_edge_length(vertices, edges):
    """
    Compute the length of each edge in a vectorized manner.
    
    vertices: (num_vertices, 3) numpy array
    edges: (num_edges, 2) numpy array
    """
    # Get the two sets of vertices for each edge
    v1 = vertices[edges[:, 0]]  # shape: (num_edges, 3)
    v2 = vertices[edges[:, 1]]  # shape: (num_edges, 3)
    
    # Compute the difference between each pair of vertices
    edge_vectors = v1 - v2  # shape: (num_edges, 3)

    # Compute the Euclidean length of each edge (L2 norm) across the last axis
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    return edge_lengths


def compute_dihedral_angle(vertices, faces, edges, faces_per_edge):
    """
    Compute the dihedral angle for each edge in a vectorized manner.
    
    vertices: (num_vertices, 3) numpy array
    faces: (num_faces, 3) numpy array
    edges: (num_edges, 2) numpy array
    faces_per_edge: (num_edges, 2) numpy array where each row contains the two adjacent faces for an edge
                    If the edge is on the boundary, the second face should be -1 or an invalid index.
    """
    # Filter edges that have two adjacent faces (i.e., not boundary edges)
    valid_edges_mask = (faces_per_edge[:, 0] >= 0) & (faces_per_edge[:, 1] >= 0)
    
    # Get the valid edges and their corresponding faces
    valid_faces_1 = faces[faces_per_edge[valid_edges_mask, 0]]
    valid_faces_2 = faces[faces_per_edge[valid_edges_mask, 1]]
    
    # Get the vertices for each face of the valid edges
    v1_f1 = vertices[valid_faces_1[:, 0]]
    v2_f1 = vertices[valid_faces_1[:, 1]]
    v3_f1 = vertices[valid_faces_1[:, 2]]
    
    v1_f2 = vertices[valid_faces_2[:, 0]]
    v2_f2 = vertices[valid_faces_2[:, 1]]
    v3_f2 = vertices[valid_faces_2[:, 2]]

    # Compute the normal vectors for both sets of faces
    normal_f1 = np.cross(v2_f1 - v1_f1, v3_f1 - v1_f1)
    normal_f2 = np.cross(v2_f2 - v1_f2, v3_f2 - v1_f2)

    # Normalize the normal vectors
    normal_f1 /= np.linalg.norm(normal_f1, axis=1, keepdims=True)
    normal_f2 /= np.linalg.norm(normal_f2, axis=1, keepdims=True)

    # Compute the dihedral angle using the dot product of normal vectors
    cosine_angle = np.einsum('ij,ij->i', normal_f1, normal_f2)  # Dot product for each normal pair
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Clip to avoid numerical issues

    # Calculate the dihedral angles
    dihedral_angles = np.arccos(cosine_angle)

    # For edges on the boundary, set their dihedral angle to 0 or a special value
    boundary_angles = np.zeros(np.count_nonzero(~valid_edges_mask))

    # Combine valid angles with boundary angles
    full_dihedral_angles = np.zeros(len(edges))
    full_dihedral_angles[valid_edges_mask] = dihedral_angles
    full_dihedral_angles[~valid_edges_mask] = boundary_angles

    return full_dihedral_angles

def compute_edge_features(vertices, faces):
    """
    Prepare the edge-based features for MeshCNN.
    - vertices: array of vertex positions.
    - faces: array of face indices.

    Returns edge features including:
    - edge lengths
    - dihedral angles
    """
    edges, edge_to_faces = build_edges_vectorized(faces)

    # Compute edge features
    edge_lengths = compute_edge_length(vertices, edges)
    dihedral_angles = compute_dihedral_angle(vertices, faces, edges, edge_to_faces)

    # Combine features into one array
    edge_features = np.stack([edge_lengths, dihedral_angles], axis=-1)

    return edge_features