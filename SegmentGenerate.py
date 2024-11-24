import os
import torch
import numpy as np
from factories.model_factory import get_model
from factories.sampling_factory import get_sampling_technique


def SegmentGenerate(vertices_np, jaw=0, model_name="DynamicGraphCNN", mode="segmentation", pretrained="/path/to/checkpoint.pth", 
            sample=True, clean=True, device=torch.device("cuda:0")):
    """
    Segments 3D vertices using a pre-trained model for segmentation.

    Args:
        vertices_np (np.ndarray): Input vertices of shape (num_points, 3).
        jaw (int): Identifier for the jaw (0 or 1).
        model_name (str): The architecture of the segmentation model. 
                          Options: "DynamicGraphCNN", "PCT", "PointNet", "Folding".
        pretrained (str): Path to the pre-trained model checkpoint.
        mode (str): Specify the task.
                           Options: "segmentation", "classification", "generation"
        sample (bool): Whether to downsample the vertices using a sampling technique.
        clean (bool): Whether to clean the vertices by removing outliers.
        device (torch.device): Device to run the model on (e.g., CPU or CUDA).

    Returns:
        np.ndarray: Segmentation output of shape (1, num_points).
        OR
        np.ndarray: Generation output of shape (1, Out_num_points, 3).
    """
    # Load the model using a factory
    if mode == "generation":
        model = get_model("FoldingNet").to(device)
    else:
        model = get_model(model_name, mode="segmentation", k=33).to(device)

    # Load pretrained weights if provided
    if pretrained and os.path.exists(pretrained):
        try:
            print(f"Loading pretrained model from {pretrained}")
            state_dict = torch.load(pretrained, map_location=device)

            # Strip "module." prefix if the model was saved with DataParallel
            state_dict = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Failed to load pretrained model from {pretrained}. Error: {e}")
            print(f"Ensure the checkpoint matches the architecture '{model_name}'.")
    else:
        print(f"No pretrained weights found. Initializing a new '{model_name}' model.")

    model.eval()

    # Clean the input vertices to remove outliers
    if clean:
        print(vertices_np.shape)
        origin = np.mean(vertices_np, axis=0)

        z_values = vertices_np[:, 2]
        y_values = vertices_np[:, 1]
        x_values = vertices_np[:, 0]

        y_mean, y_std = np.mean(y_values), np.std(y_values)
        x_mean, x_std = np.mean(x_values), np.std(x_values)
        alpha = 2.0  # Threshold factor

        valid_mask = (
            (z_values > (origin[2] - 5)) &
            (y_values < (y_mean + alpha * y_std)) & (y_values > (y_mean - alpha * y_std)) &
            (x_values < (x_mean + alpha * x_std)) & (x_values > (x_mean - alpha * x_std))
        )
        vertices_np = vertices_np[valid_mask]

    # Downsample the vertices if sampling is enabled
    if sample:
        sampling = get_sampling_technique("fpsample")
        vertices_np, _ = sampling(vertices_np, 4096, 2)

    # Convert vertices and jaw to tensors
    vertices = torch.tensor(vertices_np, dtype=torch.float32, device=device).view(-1, 3).unsqueeze(0)
    jaw = torch.tensor(jaw % 2, dtype=torch.long, device=device).reshape(-1)

    # Perform segmentation using the model
    if mode != "generation":
        output = model(vertices, jaw)[0]
        output = torch.max(output, dim=2)[1].cpu().detach().numpy()
    else:
        output = model(vertices)

    return output
