"""
Main entry point for running the retrieval comparison module as a script.

Usage:
    python -m retrieval_comparison --help
    python -m retrieval_comparison demo --output ./demo_output
"""

from .cli import main

if __name__ == "__main__":
    main()
