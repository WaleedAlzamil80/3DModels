import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from factories.dataset_factory import get_dataset_loader
from factories.model_factory import get_model
from metrics.meanAccClass import compute_mean_per_class_accuracy
from metrics.mIOU import compute_mIoU
from rigidTransformations import apply_random_transformation
from config.args_config import parse_args


# Helper functions
def plot_confusion_matrix(cm, classes, title="Confusion Matrix", output_path="confusion_matrix.png"):
    """
    Plot and save the confusion matrix.

    Parameters:
    cm (numpy.ndarray): Confusion matrix.
    classes (list): List of class names.
    title (str): Title of the plot.
    output_path (str): Path to save the plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_test_statistics(report):
    """
    Print detailed statistics from the classification report.

    Parameters:
    report (dict): Classification report as a dictionary.
    """
    print("\nDetailed Test Statistics:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"Class {label}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

    overall = {k: v for k, v in report.items() if k in ["accuracy", "macro avg", "weighted avg"]}
    print("\nOverall Metrics:")
    for metric_name, metrics in overall.items():
        if isinstance(metrics, dict):
            print(f"{metric_name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{metric_name}: {metrics:.4f}")

# Parse arguments
args = parse_args()

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

# Load test set
def load_test_data(args):
    _, test_loader = get_dataset_loader(args.Dataset, args)
    return test_loader

# Load model
def load_model(args):
    model = get_model(args.model, mode=args.mode, k=args.k).to(device)
    model = torch.nn.DataParallel(model).to(device)

    if args.pretrained is not None and os.path.exists(args.pretrained):
        print(f"Loading pretrained model from {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
    else:
        raise FileNotFoundError(f"Pretrained model not found at {args.pretrained}")

    model.eval()
    return model

def test_model(model, test_loader, args):
    test_labels = []
    test_preds = []
    test_loss = 0

    criterion = torch.nn.CrossEntropyLoss()
    test_miou = []
    test_acc = []

    with torch.no_grad():
        for vertices, labels, jaw in tqdm(test_loader, desc="Testing"):
            vertices, labels, jaw = vertices.to(device), labels.to(device), jaw.to(device)

            if args.rigid_augmentation_test:
                vertices = apply_random_transformation(vertices, rotat=args.rotat)

            outputs = model(vertices, jaw)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 2)
            
            test_miou.append(compute_mIoU(preds.reshape(-1, args.n_centroids * args.nsamples),
                                          labels.reshape(-1, args.n_centroids * args.nsamples), args.k))
            test_acc.append(compute_mean_per_class_accuracy(preds.reshape(-1, args.n_centroids * args.nsamples),
                                                            labels.reshape(-1, args.n_centroids * args.nsamples), args.k))

            test_labels.extend(labels.view(-1).cpu().numpy())
            test_preds.extend(preds.view(-1).cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    avg_miou = np.mean(test_miou)
    avg_acc = np.mean(test_acc)

    return test_labels, test_preds, avg_loss, avg_miou, avg_acc

def generate_statistics(test_labels, test_preds, avg_loss, avg_miou, avg_acc, args):
    # Classification Report
    report = classification_report(test_labels, test_preds, output_dict=True)
    print("Classification Report:")
    print(classification_report(test_labels, test_preds))

    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, args.output, classes=np.arange(args.k),
                          title=f"Confusion Matrix for {args.model}")

    # Summary of Metrics
    print_test_statistics(avg_loss, avg_miou, avg_acc, report)

    # Save the report
    report_path = os.path.join(args.output, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(classification_report(test_labels, test_preds))

    # Visualization placeholder
    print("Generating visualizations...")
    # Add any other custom visualization logic


def test_model(model, test_loader, args):
    """
    Function to evaluate a trained model on the test dataset and generate statistics and visualizations.

    Parameters:
    model (torch.nn.Module): Trained model to evaluate.
    test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
    args (argparse.Namespace): Command-line arguments.
    """
    cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if cuda else 'cpu'

    model = model.to(device)
    model.eval()

    test_labels = []
    test_preds = []

    with torch.no_grad():
        for vertices, labels, jaw in tqdm(test_loader, desc="Testing"):
            vertices, labels, jaw = vertices.to(device), labels.to(device), jaw.to(device)

            # Forward pass
            outputs = model(vertices, jaw)

            # Get predictions
            _, preds = torch.max(outputs, 2)
            test_labels.extend(labels.view(-1).cpu().numpy())
            test_preds.extend(preds.view(-1).cpu().numpy())

    # Calculate statistics
    print("Generating classification report...")
    report = classification_report(test_labels, test_preds, output_dict=True)
    print_test_statistics(report)

    # Generate confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(test_labels, test_preds)
    cm_path = os.path.join(args.output, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes=args.class_names, title="Confusion Matrix", output_path=cm_path)

    # Save metrics to file
    metrics_path = os.path.join(args.output, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(classification_report(test_labels, test_preds))

    print(f"Test metrics and confusion matrix saved in {args.output}")

    return report, cm

if __name__ == "__main__":
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    test_loader = load_test_data(args)
    model = load_model(args)
    test_labels, test_preds, avg_loss, avg_miou, avg_acc = test_model(model, test_loader, args)
    generate_statistics(test_labels, test_preds, avg_loss, avg_miou, avg_acc, args)
