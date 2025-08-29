import os
import yaml
    
def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def load_tickers_from_yaml(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Accept either a plain list, or a mapping with a 'tickers' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and isinstance(data.get("tickers"), list):
        return data["tickers"]
    else:
        raise ValueError("YAML must be a list of tickers or a mapping with a 'tickers' list.")