from pathlib import Path
import yaml

def load_yaml(path):
    with Path(path).open() as stream: return yaml.safe_load(stream)
