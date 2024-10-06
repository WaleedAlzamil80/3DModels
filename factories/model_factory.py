from models.PointNetpp.PointNet2d import PointNet
from models.PointNetpp.PointNetPP import PointNetpp
from models.GraphCNN.DGCNN import DGCNN

# Dictionary that maps model names to model classes
MODEL_FACTORY = {
    "PointNet": PointNet,
    "PointNet++": PointNetpp,
    # "FoldingNet": FoldingNet,
    "DynamicGraphCNN": DGCNN,
    # "MeshCNN": MeshCNN,
    # "PCT": PCTransformer,
}

def get_model(name, **kwargs):
    """Fetch the model from the factory."""
    if name not in MODEL_FACTORY:
        raise ValueError(f"Model {name} is not available.")
    return MODEL_FACTORY[name](**kwargs)