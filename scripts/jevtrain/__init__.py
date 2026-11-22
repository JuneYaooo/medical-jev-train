"""Source checkout compatibility; install with pip install -e . for the jev command."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[2] / "src/jevtrain")]
