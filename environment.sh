#!/bin/bash

# Step 1: Set environment name
ENV_NAME="3D_Models"

# Step 2: Check if the environment already exists
if [ -d "$ENV_NAME" ]; then
  echo "Virtual environment '$ENV_NAME' already exists."
else
  # Step 3: Create the virtual environment
  echo "Creating virtual environment..."
  python3 -m venv $ENV_NAME

  # Step 4: Check if creation was successful
  if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    exit 1
  fi
  echo "Virtual environment created successfully."
fi

# Step 5: Activate the virtual environment
echo "Activating virtual environment..."
source "$ENV_NAME/bin/activate"

# Step 6: Install dependencies
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "No requirements.txt file found. Please add it and rerun the script."
fi

# Step 7: Indicate successful setup
echo "Environment setup complete. Virtual environment '$ENV_NAME' is activated."

# The environment will remain active after the script finishes.
# To deactivate it, run 'deactivate' in the terminal manually.