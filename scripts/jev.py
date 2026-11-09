"""Compatibility entry point; installed users can run jev."""
from jevtrain.cli import main
from jevtrain.config import load_config

if __name__ == "__main__":
    main()
