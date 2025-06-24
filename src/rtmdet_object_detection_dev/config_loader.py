from pathlib import Path

import yaml


def convert_dict(d: dict) -> None:
    """Convert the dict values to suitable types."""
    for k, v in d.items():
        if isinstance(v, dict):
            convert_dict(v)
            continue
        if isinstance(v, list):
            for item in filter(lambda x: isinstance(x, dict), v):
                convert_dict(item)
            continue
        if isinstance(v, str) and k.endswith("path"):
            d[k] = Path(v)
            continue
        if k == "config" and v is None:
            d[k] = {}


def load_yaml_file(yaml_file_path: Path) -> dict:
    """Load and convert config from a yaml file."""
    with yaml_file_path.open() as f:
        config = yaml.safe_load(f)

    convert_dict(config)

    return config
