from models.PointNetpp.PointNet2d import PointNet, TNetkd
from models.PointNetpp.PointNetPP import PointNetpp
from models.PointNetpp.SPwGKL import TNetkd as TNmod
from models.GraphCNN.DGCNN import DGCNN

from models.FoldingNet.Mining import GaussianKernelConv
from models.FoldingNet.FoldingNet import FoldingNet

from models.Transformers.PCT import PCTransformer

# Dictionary that maps model names to model classes
MODEL_FACTORY = {
    "PointNet": PointNet,
    "PointNet++": PointNetpp,
    "KCNet": GaussianKernelConv,
    "FoldingNet": FoldingNet,
    "DynamicGraphCNN": DGCNN,
    "SpatialTransformer": TNetkd,
    "SpatialTransformer_v2": TNmod,
    "PCT": PCTransformer,
    "PCTGeometryAware":PCTransformer,
}

def get_model(name, **kwargs):
    """Fetch the model from the factory."""
    if name not in MODEL_FACTORY:
        raise ValueError(f"Model {name} is not available.")
    return MODEL_FACTORY[name](**kwargs)
