#!/usr/bin/env python3
"""
Image optimizer tool that recursively optimizes PNG images in place.
Usage: uv run python3 tools/ImageOptimizer.py <directory> [--level=4]
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import oxipng

from com_blockether_catalyst.knowledge.ImageOptimizer import ImageOptimizer

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Recursively optimize PNG images in place'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Directory to recursively search for PNG images'
    )
    parser.add_argument(
        '--level',
        type=int,
        default=ImageOptimizer.DEFAULT_OPTIMIZATION_LEVEL,
        choices=range(7),
        help='Optimization level (0-6, default: 4). Higher = better compression but slower'
    )

    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        exit(1)

    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        exit(1)

    optimizer = ImageOptimizer(directory, level=args.level)
    optimizer.optimize()


if __name__ == '__main__':
    main()
