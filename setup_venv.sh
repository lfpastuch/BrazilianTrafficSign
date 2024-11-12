#!/bin/bash

# Define the path to the virtual environment and activation script
VENV_DIR="venv"
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"

# Remove the existing virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment in $VENV_DIR..."
    rm -rf "$VENV_DIR"
fi

# Create a new Python virtual environment
echo "Creating Python virtual environment in $VENV_DIR..."
python3.8 -m venv "$VENV_DIR"

# Activate the virtual environment
source "$ACTIVATE_SCRIPT"

# Upgrade pip and install necessary Python packages
echo "Upgrading pip..."
python3.8 -m pip install --upgrade pip

echo "Installing required packages from requirements.txt..."
python3.8 -m pip install -r requirements.txt

# Deactivate the virtual environment
deactivate

echo "Venv setup completed successfully."