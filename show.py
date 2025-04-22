import argparse
import torch
import numpy as np
import os
import open3d as o3d
import imageio
import fastmesh as fm
from prepare_vertices import preprocess

device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

from models.Transformers.PCT import PCTransformer
model = PCTransformer("segmentation").to(device)

def load_model(model_path):
    # Load the segmentation model
    state_dict = torch.load(model_path, map_location=device)

    # Strip "module." prefix if the model was saved with DataParallel
    state_dict = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    model.eval()
    return model

def visualize_segmentation(model, input_pointcloud_path, output_path, format):
    # Load the 3D point cloud
    points = fm.load(input_pointcloud_path)[0]
    points, _ = preprocess(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Convert point cloud to tensor
    input_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)

    # Perform segmentation
    with torch.no_grad():
        output = model(input_tensor, torch.tensor(0, dtype=torch.long, device=device).reshape(-1).to(device))
        segmentation = torch.argmax(output.squeeze(), dim=1).detach().cpu().numpy()

    # Assign colors based on segmentation labels
    unique_labels = np.unique(segmentation)
    label_colors = {label: np.random.rand(3) for label in unique_labels}
    segmented_colors = np.array([label_colors[label] for label in segmentation])

    # Update point cloud colors
    pcd.colors = o3d.utility.Vector3dVector(segmented_colors)

    # Save visualization as GIF or video
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)

    if format == "gif":
        gif_path = os.path.join(output_path, "segmentation.gif")
        frames = []
        for angle in range(0, 360, 10):
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            frames.append((np.asarray(image) * 255).astype(np.uint8))
        imageio.mimsave(gif_path, frames, duration=0.1)
    elif format == "video":
        video_path = os.path.join(output_path, "segmentation.mp4")
        writer = imageio.get_writer(video_path, fps=10)
        for angle in range(0, 360, 10):
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            writer.append_data((np.asarray(image) * 255).astype(np.uint8))
        writer.close()
    else:
        raise ValueError("Invalid format. Use 'gif' or 'video'.")

    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Visualize Segmentation Model on 3D Point Cloud")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the segmentation model")
    parser.add_argument("--input_pointcloud", type=str, required=True, help="Path to the input 3D point cloud")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output")
    parser.add_argument("--format", type=str, choices=["gif", "video"], required=True, help="Output format (gif or video)")
    args = parser.parse_args()

    model = load_model(args.model_path)
    visualize_segmentation(model, args.input_pointcloud, args.output_path, args.format)

if __name__ == "__main__":
    main()

# python -m scripts.visualize_segmentation --model_path trials/25-04-07/PCT_50.pth --output_path {output_path} --format {gif_or_video}