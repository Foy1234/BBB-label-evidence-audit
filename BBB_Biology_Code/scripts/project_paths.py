import os
from pathlib import Path

def project_root():
    value = os.environ.get("BBB_PROJECT_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[1]

def project_path(*parts):
    return project_root().joinpath(*parts)
