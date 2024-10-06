from .pointnet import train as pointnetTrain
from .pointnetpp import train as pointnetPPTrain
from .DynamicGraphCNN import train as graphTrain
from .MeshCNN import train as meshTrain

# Mode Factory that maps modes to classes
TRAIN_FACTORY = {
    "PointNet": pointnetTrain,
    "PointNet++": pointnetPPTrain,
    "DynamicGraphCNN": graphTrain,
    "MeshCNN": meshTrain,
}

def get_train(model, *args, **kwargs):
    """Fetch the appropriate PointNet model based on the mode."""
    if model not in TRAIN_FACTORY:
        raise ValueError(f"Mode {model} is not available.")
    return TRAIN_FACTORY[model](*args, **kwargs)
