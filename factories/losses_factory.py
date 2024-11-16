from losses.Dice_Focal_losses import DiceLoss, FocalLoss
from losses.evalMetrics.chamferDistance import ChamferLoss
from losses.evalMetrics.HausdorffDistance import HausdorffLoss
from torch.nn import CrossEntropyLoss

# Dictionary that maps model names to model classes
LOSS_FACTORY = {
    "hausdorff": HausdorffLoss,
    "chamfer": ChamferLoss,
    "focal": FocalLoss,
    "dice": DiceLoss,
    "crossentropy": CrossEntropyLoss,
}

def get_loss(name, **kwargs):
    """Fetch the Loss from the factory."""
    if name not in LOSS_FACTORY:
        raise ValueError(f"Loss {name} is not available.")
    return LOSS_FACTORY[name](**kwargs)
