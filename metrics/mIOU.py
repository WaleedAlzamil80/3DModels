import torch

def compute_mIoU(preds, labels, num_classes):
    """
    Calculate the mean Intersection over Union (mIoU) for multi-class segmentation.

    Args:
        preds (torch.Tensor): Predicted labels, shape (batch_size, num_points).
        labels (torch.Tensor): True labels, shape (batch_size, num_points).
        num_classes (int): Number of classes.

    Returns:
        float: The mean IoU across all classes.
    """
    ious = []
    for cls in range(num_classes):
        # True positives (TP): Predicted as class `cls` and actually belongs to class `cls`
        TP = torch.sum((preds == cls) & (labels == cls)).float()
        
        # False positives (FP): Predicted as class `cls` but actually belongs to a different class
        FP = torch.sum((preds == cls) & (labels != cls)).float()
        
        # False negatives (FN): Actually belongs to class `cls` but predicted as a different class
        FN = torch.sum((preds != cls) & (labels == cls)).float()

        # IoU for the class
        denominator = TP + FP + FN
        iou = TP / denominator if denominator != 0 else torch.tensor(0.0).cuda()
        ious.append(iou)

    # Calculate mean IoU by averaging across all classes
    mIoU = torch.mean(torch.stack(ious))
    return mIoU.item()
