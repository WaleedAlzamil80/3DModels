from .train_pointnet import train as pointnetTrain
from .train_dgcnn import train as graphTrain

# Mode Factory that maps modes to classes
TRAIN_FACTORY = {
    "PointNet": pointnetTrain,
    "PointNet++": pointnetTrain,
    "DynamicGraphCNN": graphTrain,
    "MeshCNN": pointnetTrain,
}

def get_train(model, *args, **kwargs):
    """Fetch the appropriate PointNet model based on the mode."""
    if model not in TRAIN_FACTORY:
        raise ValueError(f"Mode {model} is not available.")
    return TRAIN_FACTORY[model](*args, **kwargs)
