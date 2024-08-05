---

# Titanic Data Processing and Modeling

## Overview

This project demonstrates a pipeline for processing Titanic dataset features, training a logistic regression model, and evaluating its performance. The workflow involves data preprocessing, feature engineering, data transformation, model training, and evaluation. Additionally, the setup for environment preparation and dataset handling is provided.

## Project Structure

- `src/dbms_7.py`: Python script containing data preprocessing, feature engineering, transformation, and model training.
- `setup.sh`/`setup.bat`: Scripts for setting up the environment, downloading the dataset, and running the Python script.
- `requirements.txt`: Contains the necessary Python package dependencies.

## Prerequisites

Ensure you have the following installed:
- Python 3.x
- `pip` for Python package management
- `kaggle` CLI for downloading datasets

## Setup

### `setup.sh`

For Unix-based systems, use the `setup.sh` script:

```bash
#!/bin/bash

# Install the required packages
pip install -r requirements.txt

# Create data folder if it doesn't exist
mkdir -p data

# Download dataset to data folder
kaggle competitions download -c titanic -p data

# Unzip dataset using Python
python -c "import zipfile; zipfile.ZipFile('data/titanic.zip', 'r').extractall('data')"

# Delete the zip file to keep the folder clean
rm data/titanic.zip

# Running the script
python -m src.dbms_7
```

### `setup.bat`

For Windows systems, use the `setup.bat` script:

```batch
@echo off
setlocal

REM Install the required packages
python -m pip install -r requirements.txt

REM Create data folder if it doesn't exist
if not exist data (
    mkdir data
)

REM Download dataset to data folder
kaggle competitions download -c titanic -p data

REM Unzipping dataset using Python
python -c "import zipfile; zipfile.ZipFile('data/titanic.zip', 'r').extractall('data')"

REM Delete the zip file to keep the folder clean
del data\titanic.zip

REM Running the script
python -m src.dbms_7

endlocal
```

## Python Script: `src/dbms_7.py`

### Dependencies

- `pandas`
- `sklearn`
- `sqlalchemy`
- `warnings`
- `logging`

### Functionality

1. **Data Loading**:
   - Reads training and test datasets from CSV files.

2. **Data Preprocessing**:
   - Handles missing values in 'Age', 'Embarked', and 'Fare' columns.
   - Drops the 'Cabin' column.

3. **Feature Engineering**:
   - Extracts titles from 'Name' and creates a 'FamilySize' feature.

4. **Data Transformation**:
   - Scales numerical features ('Age', 'Fare', 'FamilySize').
   - One-hot encodes categorical features ('Sex', 'Embarked', 'Title').

5. **Database Operations**:
   - Saves transformed datasets into an SQLite database.
   - Loads the data from the database for model training and prediction.

6. **Model Training and Evaluation**:
   - Trains a logistic regression model on the training set.
   - Evaluates model accuracy on a validation set and, if applicable, on the test set.

### Error Handling

Logs errors encountered during execution.

## Running the Project

1. Set up the environment and download the dataset by running the appropriate `setup.sh` or `setup.bat` script.
2. The `src/dbms_7.py` script will be executed by the appropriate `setup.sh` or `setup.bat`.

---
