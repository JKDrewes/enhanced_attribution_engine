"""Provides project directory paths and ensures necessary directories exist. 
That way a developer can run any of the main project scripts 
as entrypoints without needing to first run a separate setup step.
This allows for modular testing from VSCode."""

import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "config"

# Commonly accessed data paths
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Ensure critical directories exist
for d in (SRC_DIR, DATA_DIR, CONFIG_DIR, RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR):
    d.mkdir(exist_ok=True, parents=True)

# Ensure matplotlib has a config dir (avoids Path.home() issues on some systems)
mpl_config_dir = ROOT_DIR / ".matplotlib"
mpl_config_dir.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

# Add both project root and src/ to Python path if not already there
for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Export common paths so other modules can use them
__all__ = [
    'ROOT_DIR',
    'SRC_DIR',
    'DATA_DIR',
    'CONFIG_DIR',
    'RAW_DIR',
    'PROCESSED_DIR',
    'OUTPUTS_DIR',
]
