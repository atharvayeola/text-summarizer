import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format = '[%(asctime)s]:%(message)s:')

project_name = "textSummarizer"
files = [
    ".github/worflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb"
]

for fpath in files:
    fpath = Path(fpath)
    fdir, fname = os.path.split(fpath)

    if fdir!="":
        os.makedirs(fdir, exist_ok=True)
        logging.info(f"Created directory: {fdir} for the file {fname}")
    if (not os.path.exists(fpath) or os.path.getsize(fpath) == 0):    
        with open(fpath, 'w') as f:
            pass
            logging.info(f"Created empty file: {fpath}")

    else:
        logging.info(f"File {fpath} already exists.") 