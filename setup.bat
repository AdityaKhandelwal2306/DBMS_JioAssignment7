@echo off
setlocal

REM Install the required packages
python -m pip install -r requirements.txt

REM Create data folder if it doesn't exist
if not exist data (
    mkdir data
)

REM Downloading dataset to data folder
kaggle competitions download -c titanic -p data

REM Unzipping dataset using Python
python -c "import zipfile; zipfile.ZipFile('data/titanic.zip', 'r').extractall('data')"

REM Delete the zip file to keep the folder clean
del data\titanic.zip

REM Running the script
python -m src.dbms_7


endlocal
