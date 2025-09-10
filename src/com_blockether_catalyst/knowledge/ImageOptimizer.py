import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import oxipng


class ImageOptimizer:
    """Optimizes PNG images recursively in a directory."""

    DEFAULT_WORKER_COUNT = 4
    DEFAULT_OPTIMIZATION_LEVEL = 4

    def __init__(
        self,
        directory: Path,
        max_workers: Optional[int] = os.cpu_count(),
        level: int = DEFAULT_OPTIMIZATION_LEVEL,
    ):
        """
        Initialize the ImageOptimizer.

        Args:
            directory: Root directory to search for images
            level: Optimization level (0-6, higher = better compression but slower)
        """
        if not directory.exists():
            raise ValueError(f"ImageOptimizer: Directory '{directory}' does not exist")

        if level < 0 or level > 6:
            raise ValueError("ImageOptimizer: Optimization level must be between 0 and 6")

        self._directory = directory
        self._level = level
        self._worker_count = max_workers or self.DEFAULT_WORKER_COUNT

    def _find_png_files(self) -> List[Path]:
        """Find all PNG files recursively in the directory."""
        return list(self._directory.rglob("*.png"))

    def _optimize_single_png_file(self, file_path: Path) -> None:
        """
        Optimize a single PNG file in place.

        Args:
            file_path: Path to the PNG file to optimize
        """
        try:
            oxipng.optimize(str(file_path), str(file_path), level=self._level)
            relative_path = file_path.relative_to(self._directory)
            print(f"ImageOptimizer: Optimized: {relative_path}")
        except oxipng.PngError as e:
            print(f"ImageOptimizer: Failed (PNG Error): {file_path} - {e}")
        except Exception as e:
            print(f"ImageOptimizer: Error processing {file_path}: {e}")

    def optimize(self) -> None:
        """Optimize all PNG files found in the directory tree."""
        png_files = self._optimize_all_png_files()

        print(f"ImageOptimizer: Batch optimization complete. Processed {len(png_files)} files.")

    def _optimize_all_png_files(self):
        png_files = self._find_png_files()

        print(f"ImageOptimizer: Found {len(png_files)} PNG files to optimize")
        print(f"ImageOptimizer: Using optimization level: {self._level}")
        print(f"ImageOptimizer: Using {self._worker_count} worker threads\n")

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self._worker_count) as executor:
            list(executor.map(self._optimize_single_png_file, png_files))
        return png_files
