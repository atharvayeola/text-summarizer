import importlib
import os
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from pathlib import Path
from typing import Any, Callable, List
import yaml
from textSummarizer.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return its content as a ConfigBox."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as exc:
        raise ValueError("yaml file is empty") from exc
    except Exception as exc:
        raise exc


@ensure_annotations
def create_directories(path_to_directories: List[Path], verbose: bool = True) -> None:
    """Create the provided directories if they do not exist."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """Return the file size in kilobytes."""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def import_from_string(dotted_path: str) -> Callable[..., Any]:
    """Import a callable using a dotted module path."""
    module_path, _, attribute = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Unable to import '{dotted_path}'. Provide a full dotted path.")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(f"Module '{module_path}' does not define '{attribute}'.") from exc
