from pathlib import Path
import joblib

def ensure_dirs(*dirs):
    for d in dirs: Path(d).mkdir(parents=True,exist_ok=True)

def save_object(obj,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True); joblib.dump(obj,path)
