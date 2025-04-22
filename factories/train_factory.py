from train.train_pointnet import train as pointnetTrain
from train.train_dgcnn import train as graphTrain
from train.train_sp import train as sptrain
from train.train_pct import train as pctTrain

# Factory to choose the suitable training Loop use args
TRAIN_FACTORY = {
    "PointNet": pointnetTrain,
    "PointNet++": pointnetTrain,
    "KCNet": graphTrain,
    "FoldingNet": graphTrain,
    "SpatialTransformer": sptrain,
    "SpatialTransformer_v2": sptrain,
    "SpatialTransformer_v3": sptrain,
    "DynamicGraphCNN": graphTrain,
    "PCT": pctTrain,
}

def get_train(model, *args, **kwargs):
    """Fetch the appropriate PointNet model based on the mode."""
    if model not in TRAIN_FACTORY:
        raise ValueError(f"Mode {model} is not available.")
    return TRAIN_FACTORY[model](*args, **kwargs)
