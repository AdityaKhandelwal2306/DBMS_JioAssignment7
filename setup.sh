#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Install the required packages
python -m pip install -r requirements.txt

# Create data folder if it doesn't exist
if [ ! -d "data" ]; then
    mkdir data
fi

# Downloading dataset to data folder
kaggle competitions download -c titanic -p data

# Unzipping dataset using Python
python -c "import zipfile; zipfile.ZipFile('data/titanic.zip', 'r').extractall('data')"

# Delete the zip file to keep the folder clean
rm data/titanic.zip

# Running the script
python -m src.dbms_7
