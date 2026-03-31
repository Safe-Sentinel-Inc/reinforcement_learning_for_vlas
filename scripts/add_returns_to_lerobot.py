"""Entry point for computing and appending return labels to a LeRobot dataset.

Delegates to scripts.labeling.cli which handles argument parsing and execution.
"""
from scripts.labeling.cli import main

if __name__ == "__main__":
    main()
