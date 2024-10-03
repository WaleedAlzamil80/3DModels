import os
import zipfile
import wget
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")
    parser.add_argument('--version', type=str, default="ModelNet10", help="Which dataset to Download")
    parser.add_argument('--path', type=str, default="datasetMO", help="Extract to")

    return parser.parse_args()

args = parse_args()

# Step 1: Download the dataset
def download_dataset(url, output_path):
    wget.download(url, out=output_path)
    print(f"\nDownloaded {output_path}")

# Step 2: Extract the dataset
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

# Step 3: Verify extraction
def verify_extraction(dataset_dir):
    for root, dirs, files in os.walk(dataset_dir):
        print(f"Directory: {root}")
        for file in files:
            print(f"File: {file}")
        break  # To avoid printing all subdirectories and files

# URLs for the datasets
urls = {
    'ModelNet10': "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    'ModelNet40': "http://modelnet.cs.princeton.edu/ModelNet40.zip"
}

# Define where to save the datasets
output_dir = args.path
os.makedirs(output_dir, exist_ok=True)

# Download, extract, and verify each dataset
if args.version in urls.keys():
    output_path = os.path.join(output_dir, f"{args.version}.zip")
    extract_to = os.path.join(output_dir, args.version)

    # Download
    download_dataset(urls[args.version], output_path)

    # Extract
    extract_zip(output_path, extract_to)

    # Verify
    verify_extraction(extract_to)
