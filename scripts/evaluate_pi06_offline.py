"""Entry point for running offline evaluation of a trained pi0 policy.

Delegates to scripts.evaluation.cli which handles argument parsing and execution.
"""
from scripts.evaluation.cli import main

if __name__ == "__main__":
    main()
