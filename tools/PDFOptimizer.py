#!/usr/bin/env python3
"""
PDF optimizer tool that recursively optimizes PDF files in place.
Usage: uv run python3 tools/PDFOptimizer.py <directory> [--level=4]
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from blockether_catalyst.knowledge.optimization.PDFOptimizer import PDFOptimizer


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Recursively optimize PNG images in place")
    parser.add_argument("directory", type=str, help="Directory to recursively search for PNG images")
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        exit(1)

    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        exit(1)

    optimizer = PDFOptimizer(directory)
    optimizer.optimize()


if __name__ == "__main__":
    main()
