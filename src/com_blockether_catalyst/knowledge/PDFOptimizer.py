import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader, PdfWriter


class PDFOptimizer:
    """Optimizes PDF files recursively in a directory."""

    DEFAULT_WORKER_COUNT = 4

    def __init__(self, directory: Path, max_workers: Optional[int] = os.cpu_count()):
        """
        Initialize the PDFOptimizer.

        Args:
            directory: Root directory to search for images
        """
        if not directory.exists():
            raise ValueError(f"PDFOptimizer: Directory '{directory}' does not exist")

        self._directory = directory
        self._worker_count = max_workers or self.DEFAULT_WORKER_COUNT

    def _find_pdf_files(self) -> List[Path]:
        """Find all PDF files recursively in the directory."""
        return list(self._directory.rglob("*.pdf"))

    def _optimize_single_pdf_file(self, file_path: Path) -> None:
        """
        Optimize a single PDF file in place.

        Args:
            file_path: Path to the PDF file to optimize
        """
        try:
            reader = PdfReader(str(file_path))
            writer = PdfWriter(reader)

            for page in writer.pages:
                page.compress_content_streams()
                writer.add_page(page)

            with open(file_path, "wb") as output_writer:
                writer.write(output_writer)

            print(f"PDFOptimizer: Optimized: {file_path}")
        except Exception as e:
            print(f"PDFOptimizer: Error processing {file_path}: {e}")

    def optimize(self) -> None:
        """Optimize all PDF files found in the directory tree."""
        pdf_files = self._optimize_all_pdf_files()

        print(f"PDFOptimizer: Batch optimization complete. Processed {len(pdf_files)} files.")

    def _optimize_all_pdf_files(self):
        pdf_files = self._find_pdf_files()

        print(f"PDFOptimizer: Found {len(pdf_files)} PDF files to optimize")
        print(f"PDFOptimizer: Using {self._worker_count} worker threads\n")

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self._worker_count) as executor:
            list(executor.map(self._optimize_single_pdf_file, pdf_files))
        return pdf_files
