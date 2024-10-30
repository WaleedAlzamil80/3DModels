import numpy as np
from scipy.spatial.transform import Rotation as R

def random_rigid_transform(points, rotation_range=(0, 360), translation_range=(-1, 1)):
    # Generate random rotation angles within the specified range
    rotation_angles = np.random.uniform(rotation_range[0], rotation_range[1], size=3)
    
    # Create a rotation matrix from the random angles
    rotation = R.from_euler('xyz', rotation_angles, degrees=True)
    rotation_matrix = rotation.as_matrix()
    
    # Generate random translation within the specified range
    translation_vector = np.random.uniform(translation_range[0], translation_range[1], size=(1, 3))
    
    # Apply transformation
    transformed_points = np.dot(points, rotation_matrix.T) + translation_vector
    return transformed_points