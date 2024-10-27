import torch

def compute_mean_per_class_accuracy(preds, labels, num_classes):
    """
    Calculate the mean per-class accuracy for multi-class segmentation.

    Args:
        preds (torch.Tensor): Predicted labels, shape (batch_size, num_points).
        labels (torch.Tensor): True labels, shape (batch_size, num_points).
        num_classes (int): Number of classes.

    Returns:
        float: The mean per-class accuracy.
    """
    class_accuracies = []
    for cls in range(num_classes):
        # Points in class `cls` in the true labels
        mask = (labels == cls)
        
        # True positives (TP): Predicted as class `cls` and actually belongs to class `cls`
        correct_preds = torch.sum((preds == cls) & mask).float()
        
        # Total points in class `cls` (for ground truth)
        total_class_points = torch.sum(mask).float()

        # Avoid division by zero
        class_accuracy = correct_preds / total_class_points if total_class_points != 0 else torch.tensor(0.0).cuda()
        class_accuracies.append(class_accuracy)

    # Calculate mean per-class accuracy
    mean_per_class_accuracy = torch.mean(torch.stack(class_accuracies))
    return mean_per_class_accuracy.item()
