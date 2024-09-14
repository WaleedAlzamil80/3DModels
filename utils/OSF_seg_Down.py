from osfclient.api import OSF
import zipfile
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")
    parser.add_argument('--n', type=int, default=4, help="Number of Parts to Download")
    return parser.parse_args()

args = parse_args()

# Create an OSF instance
osf = OSF()

# List of project IDs
li = ["xctdy", "5cmg3", "xfcn9", "hw7bj", "n9bd7", "7az58", "2ybr4"]

for i in range(args.n):
    project = osf.project(li[i])  # The project ID from the link
    print(li[i])
    # Access the storage node where the files are located
    storage = project.storage('osfstorage')

    # Download each file in the storage
    for file in storage.files:
        print(f'Downloading {file.name}')
        # Download the zip file
        with open(file.name, 'wb') as local_file:
            file.write_to(local_file)
        
        if li[i] != "xctdy":
            # Extract the zip file
            with zipfile.ZipFile(file.name, 'r') as zip_ref:
                extract_path = os.path.splitext(file.name)[0]  # Directory name without .zip
                print(f'Extracting {file.name} to {extract_path}')
                zip_ref.extractall(extract_path)
            
            # Optionally, delete the zip file after extraction
            os.remove(file.name)
            print(f'{file.name} has been extracted and deleted.')