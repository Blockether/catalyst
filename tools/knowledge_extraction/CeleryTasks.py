"""
Celery tasks for knowledge extraction pipeline.
Each extraction step is a separate Celery task for maximum parallelization.

Pipeline Stages:
----------------
1. stage01.extract_single_pdf - Single PDF extraction
2. stage02.batch_pdf_processing - Batch PDF processing
3. stage03.document_chunking - Document chunking
4. stage04.semantic_classification - Semantic classification
5. stage05.term_extraction - Term extraction
6. stage06.term_grouping - Term grouping
7. stage07.cooccurrence_analysis - Co-occurrence analysis
8. stage08.semantic_enrichment - Semantic enrichment
9. stage09.term_linking - Term linking
10. stage10.knowledge_graph_construction - Knowledge graph construction

Each stage processes data from the previous stage, creating a complete
knowledge extraction pipeline from raw PDFs to a structured knowledge graph.
"""

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from celery import chain, chord, group
from celery.result import AsyncResult

# Disable LiteLLM's async logging to avoid event loop conflicts with Celery prefork
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_LOG_RAW_RESPONSE"] = "False"

from blockether_catalyst.knowledge.extraction.ConcreteExtractionCalls import (
    create_extraction_calls,
)
from blockether_catalyst.knowledge.extraction.internal.PDFExtractor import (
    PDFKnowledgeExtractor,
)
from blockether_catalyst.knowledge.extraction.ModelSettings import (
    ExtractionModelSettings,
)
from blockether_catalyst.knowledge.KnowledgeTypes import (
    KnowledgeExtractionItem,
    KnowledgeExtractionOutput,
    KnowledgeExtractionResult,
    KnowledgeExtractionResultWithChunks,
    KnowledgeExtractionResultWithClassifiedChunks,
    KnowledgeProcessorSettings,
    LinkedKnowledge,
    RawExtractionData,
    RawKnowledgeChunk,
    Term,
    TermCandidate,
    TermCandidateGrouped,
    TermWithLinks,
)
from tools.knowledge_extraction.CeleryApp import celery_app

logger = logging.getLogger(__name__)

# Simple logging setup - our code at INFO, libraries at WARNING
logging.getLogger("blockether_catalyst").setLevel(logging.INFO)
logging.getLogger("tools.knowledge_extraction").setLevel(logging.INFO)

# Suppress ALL noisy libraries in one go
for lib in [
    "pdfminer",
    "PIL",
    "pypdfium2",
    "instructor",
    "httpx",
    "httpcore",
    "openai",
    "anthropic",
    "urllib3",
    "requests",
    "kombu",
    "celery",
    "py.warnings",
    "litellm",  # Suppress LiteLLM logs
    "asyncio",  # Suppress asyncio event loop errors
]:
    logging.getLogger(lib).setLevel(logging.WARNING)


# Custom asyncio exception handler to suppress LiteLLM event loop errors
def suppress_litellm_errors(loop, context):
    """Suppress LiteLLM's async logging errors in Celery prefork workers."""
    exception = context.get("exception")
    if exception and isinstance(exception, RuntimeError):
        if "bound to a different event loop" in str(exception):
            # Silently ignore LiteLLM's queue/event loop conflicts
            return
    # Log other exceptions normally
    loop.default_exception_handler(context)


def run_async_with_suppressed_errors(coro):
    """
    Run async code in Celery worker with custom exception handler.

    This helper method creates a new event loop, sets the custom exception
    handler to suppress LiteLLM errors, runs the coroutine, and properly
    cleans up the loop.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(suppress_litellm_errors)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================================
# PDF EXTRACTION TASKS (Run in parallel with prefork pool)
# ============================================================================


def extract_single_pdf_direct(
    file_path: str,
    image_output_dir: str,
    knowledge_settings: KnowledgeProcessorSettings,
) -> Optional[KnowledgeExtractionItem]:
    """
    Direct function for extracting a single PDF.
    This is the actual implementation that can be called directly.
    """
    try:
        logger.info(f"[Direct] Extracting PDF: {file_path}")
        pdf_path = Path(file_path)

        if not pdf_path.exists():
            logger.error(f"PDF file not found: {file_path}")
            return None

        # Create extractor with proper parameters
        extractor = PDFKnowledgeExtractor(
            image_output_dir=Path(image_output_dir),
            knowledge_settings=knowledge_settings,
        )

        # Extract the PDF
        result = extractor.extract(pdf_path)

        # Return as extraction item
        if result:
            return KnowledgeExtractionItem(result=result)
        else:
            logger.warning(f"No content extracted from: {file_path}")
            return None

    except Exception as e:
        logger.error(f"Error extracting {file_path}: {str(e)}", exc_info=True)
        return None


@celery_app.task(
    name="stage01.extract_single_pdf",
    queue="pdf_processing",
    max_retries=3,
    autoretry_for=(IOError, OSError),  # Only retry on I/O errors
    dont_autoretry_for=(
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
    ),  # Don't retry on code errors
)
def extract_single_pdf(
    file_path: str,
    image_output_dir: str,
    knowledge_settings: KnowledgeProcessorSettings,
) -> Optional[KnowledgeExtractionItem]:
    """
    Extract a single PDF file.
    This task runs in a separate process (prefork) to avoid pypdfium2 threading issues.

    Args:
        file_path: Path to the PDF file
        image_output_dir: Directory for saving extracted images
        knowledge_settings: Settings for knowledge processing

    Returns:
        KnowledgeExtractionItem or None if processing fails
    """
    return extract_single_pdf_direct(file_path, image_output_dir, knowledge_settings)


@celery_app.task(
    name="stage01.collect_pdf_results",
    queue="pdf_processing",
)
def collect_pdf_results(
    results: List[Optional[KnowledgeExtractionItem]],
) -> KnowledgeExtractionOutput:
    """
    Callback task to collect results from parallel PDF extraction and copy source documents.

    Args:
        results: List of extraction results from parallel tasks

    Returns:
        KnowledgeExtractionOutput containing all successfully extracted PDFs
    """
    # Filter out None results (failed extractions)
    valid_results = [r for r in results if r is not None]

    # Copy source PDFs to source_documents directory
    if valid_results and valid_results[0].result:
        # Get output directory from the first result's metadata
        first_result = valid_results[0].result
        if first_result.document_metadata and first_result.document_metadata.document_path:
            # Extract output directory from document_path
            # document_path format: "{extraction_output_dir}/source_documents/{filename}"
            doc_path = Path(first_result.document_metadata.document_path)
            source_docs_dir = doc_path.parent

            if source_docs_dir.name == "source_documents":
                # Create the directory if it doesn't exist
                source_docs_dir.mkdir(parents=True, exist_ok=True)

                # Copy each successfully extracted PDF
                copied_count = 0
                for result_item in valid_results:
                    if result_item.result and result_item.result.document_filename:
                        # Try to find the original PDF file
                        filename = result_item.result.document_filename

                        # Look for the original file in common locations
                        possible_paths = [
                            Path(filename),  # If it's already an absolute path
                            Path.cwd() / filename,  # Current directory
                            Path("input") / filename,  # Common input directory
                        ]

                        src_path = None
                        for path in possible_paths:
                            if path.exists():
                                src_path = path
                                break

                        if src_path:
                            dest_path = source_docs_dir / filename
                            try:
                                shutil.copy2(src_path, dest_path)
                                logger.info(f"[Stage 01] Copied {filename} to source_documents")
                                copied_count += 1
                            except Exception as e:
                                logger.warning(f"[Stage 01] Failed to copy {filename}: {e}")
                        else:
                            logger.warning(f"[Stage 01] Could not find original file for {filename}")

                logger.info(f"[Stage 01] Copied {copied_count}/{len(valid_results)} PDF files to source_documents")

    # Package into output format
    output = KnowledgeExtractionOutput()
    output.pdf = valid_results

    logger.info(
        f"[Stage 01] Collected results: {len(valid_results)}/{len(results)} PDFs"
    )
    return output


@celery_app.task(
    name="stage01.orchestrate_pdf_extraction",
    queue="pdf_processing",
    autoretry_for=(IOError, OSError),
    dont_autoretry_for=(TypeError, ValueError, AttributeError, KeyError),
    throws=(TypeError, ValueError, AttributeError, KeyError),  # Propagate these errors
    bind=True,  # Bind to get access to self (task instance)
)
def extract_pdf_batch(
    self,
    pdf_files: List[str],
    image_output_dir: str,
    knowledge_settings: KnowledgeProcessorSettings,
) -> KnowledgeExtractionOutput:
    """
    Orchestrate parallel extraction of PDF files across multiple workers.

    Uses Celery's replace() to transform this task into a chord that
    runs PDFs in parallel without deadlock issues.

    Args:
        self: The task instance (from bind=True)
        pdf_files: List of PDF file paths
        image_output_dir: Directory for saving extracted images
        knowledge_settings: Settings for knowledge processing

    Returns:
        KnowledgeExtractionOutput containing all extracted PDFs
    """
    logger.info(
        f"[Stage 01] Orchestrating parallel extraction of {len(pdf_files)} PDFs"
    )

    # Create parallel subtasks for each PDF
    pdf_tasks = group(
        extract_single_pdf.s(pdf_path, image_output_dir, knowledge_settings)
        for pdf_path in pdf_files
    )

    # Replace this task with a chord that runs PDFs in parallel
    # This avoids the deadlock issue by not calling get() within a task
    # Note: We pass the chord signature, not its result
    raise self.replace(chord(pdf_tasks, collect_pdf_results.s()))


# ============================================================================
# DOCUMENT PROCESSING TASKS
# ============================================================================


@celery_app.task(
    name="stage02.document_chunking",
    queue="text_processing",
    autoretry_for=(IOError, OSError),
    dont_autoretry_for=(TypeError, ValueError, AttributeError, KeyError),
)
def chunk_documents_task(
    raw_extraction: KnowledgeExtractionOutput,
    model_settings: ExtractionModelSettings,
    knowledge_settings: KnowledgeProcessorSettings,
) -> List[KnowledgeExtractionResultWithChunks]:
    """
    Perform document chunking on extracted content.

    Args:
        raw_extraction: Raw extraction output from PDF processing
        model_settings: Model configuration settings
        knowledge_settings: Knowledge processor settings with correct output directory

    Returns:
        Documents with chunks
    """
    from blockether_catalyst.knowledge.extraction.ExtractionCore import (
        KnowledgeExtractionCore,
    )

    logger.info("[Stage 02] Starting document chunking")

    calls_settings = create_extraction_calls(model_settings)
    extractor = KnowledgeExtractionCore(
        calls=calls_settings,
        settings=knowledge_settings,  # Use the passed settings instead of empty settings
    )

    # Run async code with custom exception handler to suppress LiteLLM errors
    results_with_chunks = run_async_with_suppressed_errors(
        extractor._chunk_extraction(raw_extraction)
    )

    logger.info(
        f"[Stage 02] Completed chunking for {len(results_with_chunks)} documents"
    )
    return list(results_with_chunks)


@celery_app.task(
    name="stage03.semantic_classification",
    queue="text_processing",
    autoretry_for=(IOError, OSError),
    dont_autoretry_for=(TypeError, ValueError, AttributeError, KeyError),
)
def classify_chunks_task(
    results_with_chunks: List[KnowledgeExtractionResultWithChunks],
    model_settings: ExtractionModelSettings,
    knowledge_settings: KnowledgeProcessorSettings,
) -> List[KnowledgeExtractionResultWithClassifiedChunks]:
    """
    Classify chunk content for semantic typing.

    Args:
        results_with_chunks: Documents with chunks
        model_settings: Model configuration settings
        knowledge_settings: Knowledge processor settings with correct output directory

    Returns:
        Documents with classified chunks
    """
    from blockether_catalyst.knowledge.extraction.ExtractionCore import (
        KnowledgeExtractionCore,
    )

    logger.info(f"[Stage 03] Starting chunk classification for {len(results_with_chunks)} documents")

    calls_settings = create_extraction_calls(model_settings)
    extractor = KnowledgeExtractionCore(
        calls=calls_settings,
        settings=knowledge_settings,  # Use the passed settings instead of hardcoded /tmp
    )

    # Log input chunks stats
    total_chunks = sum(len(doc.chunks) for doc in results_with_chunks)
    logger.info(f"[Stage 03] Processing {total_chunks} chunks across {len(results_with_chunks)} documents")

    # Run async code with custom exception handler to suppress LiteLLM errors
    try:
        classified_results = run_async_with_suppressed_errors(
            extractor._classify_chunk_content(results_with_chunks)
        )

        # Log output chunks stats
        total_classified = sum(len(doc.chunks) for doc in classified_results)
        logger.info(f"[Stage 03] Successfully classified {total_classified} chunks")

        # Log semantic type distribution
        all_semantic_types = []
        for doc in classified_results:
            for chunk in doc.chunks:
                # After classification, chunks are guaranteed to have semantic_types
                all_semantic_types.extend(chunk.semantic_types)

        if all_semantic_types:
            from collections import Counter
            type_counts = Counter(all_semantic_types)
            logger.info(f"[Stage 03] Semantic type distribution: {dict(type_counts)}")
        else:
            logger.warning("[Stage 03] No semantic types were assigned to any chunks!")

    except Exception as e:
        logger.error(f"[Stage 03] Chunk classification failed: {e}", exc_info=True)
        raise

    logger.info("[Stage 03] Completed chunk classification")
    return list(classified_results)


@celery_app.task(
    name="stage04.term_extraction",
    queue="text_processing",
    autoretry_for=(IOError, OSError),
    dont_autoretry_for=(TypeError, ValueError, AttributeError, KeyError),
)
def extract_terms_task(
    results_with_chunks: List[KnowledgeExtractionResultWithClassifiedChunks],
    model_settings: ExtractionModelSettings,
) -> List[TermCandidate]:
    """
    Extract term candidates from documents.

    Args:
        results_with_chunks: Documents with classified chunks

    Returns:
        List of term candidates
    """
    from blockether_catalyst.knowledge.extraction.ExtractionCore import (
        KnowledgeExtractionCore,
    )

    logger.info("[Stage 04] Starting term extraction")

    calls_settings = create_extraction_calls(model_settings)
    extractor = KnowledgeExtractionCore(
        calls=calls_settings,
        settings=KnowledgeProcessorSettings(extraction_output_dir=Path("/tmp")),
    )

    # Run async code with custom exception handler to suppress LiteLLM errors
    terms = run_async_with_suppressed_errors(
        extractor._extract_terms_candidates_from_documents(results_with_chunks)
    )

    logger.info(f"[Stage 04] Extracted {len(terms)} term candidates")
    return list(terms)


@celery_app.task(
    name="stage05.term_grouping",
    queue="text_processing",
    autoretry_for=(IOError, OSError),
    dont_autoretry_for=(TypeError, ValueError, AttributeError, KeyError),
)
def group_terms_task(
    term_candidates: List[TermCandidate],
    model_settings: ExtractionModelSettings,
) -> Dict[str, TermCandidateGrouped]:
    """
    Group similar term candidates.

    Args:
        term_candidates: List of term candidates

    Returns:
        Dictionary of grouped terms
    """
    from blockether_catalyst.knowledge.extraction.ExtractionCore import (
        KnowledgeExtractionCore,
    )

    logger.info("[Stage 06] Starting term grouping")

    calls_settings = create_extraction_calls(model_settings)
    extractor = KnowledgeExtractionCore(
        calls=calls_settings,
        settings=KnowledgeProcessorSettings(extraction_output_dir=Path("/tmp")),
    )

    grouped_terms = extractor._group_term_candidates(term_candidates)

    logger.info(f"[Stage 06] Grouped into {len(grouped_terms)} unique terms")
    return grouped_terms


@celery_app.task(
    name="stage07.cooccurrence_analysis",
    queue="text_processing",
    autoretry_for=(IOError, OSError),
    dont_autoretry_for=(TypeError, ValueError, AttributeError, KeyError),
)
def enrich_cooccurrences_task(
    grouped_terms: Dict[str, TermCandidateGrouped],
    document_to_chunks_index: Dict[str, List[RawKnowledgeChunk]],
    model_settings: ExtractionModelSettings,
) -> Dict[str, TermCandidateGrouped]:
    """
    Enrich terms with co-occurrence information.

    Args:
        grouped_terms: Grouped terms
        document_to_chunks_index: Document chunk index

    Returns:
        Terms with co-occurrence data
    """
    from blockether_catalyst.knowledge.extraction.ExtractionCore import (
        KnowledgeExtractionCore,
    )

    logger.info("[Stage 07] Starting co-occurrence analysis")

    calls_settings = create_extraction_calls(model_settings)
    extractor = KnowledgeExtractionCore(
        calls=calls_settings,
        settings=KnowledgeProcessorSettings(extraction_output_dir=Path("/tmp")),
    )

    enriched_terms = extractor._enrich_with_cooccurrences(
        grouped_terms, document_to_chunks_index
    )

    logger.info("[Stage 07] Completed co-occurrence analysis")
    return enriched_terms


@celery_app.task(
    name="stage08.semantic_enrichment",
    queue="text_processing",
    autoretry_for=(IOError, OSError),
    dont_autoretry_for=(TypeError, ValueError, AttributeError, KeyError),
)
def enrich_meanings_task(
    groups: Dict[str, TermCandidateGrouped],
    document_to_chunks_index: Dict[str, List[RawKnowledgeChunk]],
    model_settings: ExtractionModelSettings,
    knowledge_settings: KnowledgeProcessorSettings,
) -> Dict[str, Term]:
    """
    Enrich terms with meaning extraction.

    Args:
        groups: Grouped terms with co-occurrences
        document_to_chunks_index: Document chunk index
        model_settings: Model configuration settings
        knowledge_settings: Knowledge processor settings with correct output directory

    Returns:
        Terms with meanings
    """
    from blockether_catalyst.knowledge.extraction.ExtractionCore import (
        KnowledgeExtractionCore,
    )

    logger.info("[Stage 08] Starting semantic enrichment")

    calls_settings = create_extraction_calls(model_settings)
    extractor = KnowledgeExtractionCore(
        calls=calls_settings,
        settings=knowledge_settings,  # Use the passed settings instead of hardcoded /tmp
    )

    # Run async code with custom exception handler to suppress LiteLLM errors
    enriched_terms = run_async_with_suppressed_errors(
        extractor._enrich_with_terms_meanings(groups, document_to_chunks_index)
    )

    logger.info("[Stage 08] Completed semantic enrichment")
    return enriched_terms


@celery_app.task(
    name="stage09.term_linking",
    queue="text_processing",
    autoretry_for=(IOError, OSError),
    dont_autoretry_for=(TypeError, ValueError, AttributeError, KeyError),
)
def link_terms_task(
    terms: Dict[str, Term],
    model_settings: ExtractionModelSettings,
) -> Dict[str, TermWithLinks]:
    """
    Create links between related terms.

    Args:
        terms: Terms with meanings

    Returns:
        Terms with links
    """
    from blockether_catalyst.knowledge.extraction.ExtractionCore import (
        KnowledgeExtractionCore,
    )

    logger.info("[Stage 09] Starting term linking")

    calls_settings = create_extraction_calls(model_settings)
    extractor = KnowledgeExtractionCore(
        calls=calls_settings,
        settings=KnowledgeProcessorSettings(
            linking_threshold=0.65,
        ),
    )

    linked_terms = extractor._link_terms(terms)

    logger.info("[Stage 09] Completed term linking")
    return linked_terms


@celery_app.task(
    name="stage10.knowledge_graph_construction",
    queue="text_processing",
    autoretry_for=(IOError, OSError),
    dont_autoretry_for=(TypeError, ValueError, AttributeError, KeyError),
)
def build_linked_knowledge_task(
    linked_terms: Dict[str, TermWithLinks],  # From previous task in chain
    results_with_chunks: List[KnowledgeExtractionResultWithClassifiedChunks],
    extraction_timestamp: int,
    model_settings: ExtractionModelSettings,
    output_dir: str,
    original_pdf_files: List[str],
    processing_duration: Optional[int] = None,
) -> Tuple[LinkedKnowledge, List[str], str]:
    """Build final LinkedKnowledge structure.

    Args:
        results_with_chunks: Documents with chunks
        linked_terms: Fully processed terms
        extraction_timestamp: Timestamp when extraction started
        processing_duration: Total processing time in seconds (computed if ``None``)

    Returns:
        Tuple of (LinkedKnowledge, pdf_files, image_output_dir) for optimization stages
    """
    from blockether_catalyst.knowledge.extraction.ExtractionCore import (
        KnowledgeExtractionCore,
    )

    # Compute duration if the caller did not supply it (e.g., when orchestrated via chords)
    if processing_duration is None:
        processing_duration = max(int(time.time() - extraction_timestamp), 0)

    logger.info("[Stage 10] Building LinkedKnowledge graph")

    calls_settings = create_extraction_calls(model_settings)
    extractor = KnowledgeExtractionCore(
        calls=calls_settings,
        settings=KnowledgeProcessorSettings(extraction_output_dir=Path(output_dir)),
    )

    # Enrich chunks with terms
    (
        results_with_chunks_and_terms,
        document_to_chunks_index_with_terms,
    ) = extractor._enrich_chunks_with_terms(results_with_chunks, linked_terms)

    # Build LinkedKnowledge
    raw_data = RawExtractionData(
        results_with_chunks=results_with_chunks_and_terms,
        terms=linked_terms,
        document_to_chunks_index=document_to_chunks_index_with_terms,
    )

    linked_knowledge = LinkedKnowledge.from_extraction_data(
        raw_data,
        extraction_timestamp=extraction_timestamp,
        processing_duration=processing_duration,
    )

    logger.info("[Stage 10] LinkedKnowledge built successfully")

    # Use original PDF file paths for optimization stages
    pdf_files = original_pdf_files

    # Get image directory from output_dir parameter (output_dir already includes knowledge_extraction)
    image_output_dir = f"{output_dir}/images"

    # Return linked knowledge with metadata for next stages
    return (linked_knowledge, pdf_files, image_output_dir)


@celery_app.task(
    name="stage11.pdf_optimization",
    queue="pdf_processing",
    autoretry_for=(IOError, OSError),
    dont_autoretry_for=(TypeError, ValueError, AttributeError, KeyError),
)
def optimize_pdfs_task(
    extraction_result: Tuple[LinkedKnowledge, List[str], str],
) -> Tuple[LinkedKnowledge, str]:
    """
    Optimize PDF files in source_documents directory.

    Args:
        extraction_result: Tuple of (LinkedKnowledge, pdf_files, image_output_dir)

    Returns:
        Tuple of (LinkedKnowledge, image_output_dir) for next stage
    """
    from blockether_catalyst.knowledge.optimization.PDFOptimizer import PDFOptimizer

    linked_knowledge, pdf_files, image_output_dir = extraction_result

    logger.info("[Stage 11] Starting PDF optimization for %d files", len(pdf_files))

    # Get source_documents directory (files should already be there from extraction stage)
    # image_output_dir is {output_dir}/images, so go up 1 level to get output_dir (which already includes knowledge_extraction)
    output_dir = Path(image_output_dir).parent
    source_docs_dir = output_dir / "source_documents"

    if not source_docs_dir.exists():
        logger.warning("[Stage 11] source_documents directory does not exist: %s", source_docs_dir)
        source_docs_dir.mkdir(parents=True, exist_ok=True)

    # Check which files are already in source_documents
    existing_files = list(source_docs_dir.glob("*.pdf"))
    logger.info("[Stage 11] Found %d PDF files in source_documents directory", len(existing_files))

    if len(existing_files) == 0:
        logger.warning("[Stage 11] No PDF files found in source_documents directory - files should have been copied during extraction")

    # Run optimization on files in source_documents
    optimizer = PDFOptimizer(directory=source_docs_dir, max_workers=4)
    optimizer.optimize()

    logger.info(
        "[Stage 11] Optimization complete for %d files in source_documents",
        len(existing_files),
    )
    return (linked_knowledge, image_output_dir)


@celery_app.task(
    name="stage12.image_optimization",
    queue="pdf_processing",
    autoretry_for=(IOError, OSError),
    dont_autoretry_for=(TypeError, ValueError, AttributeError, KeyError),
)
def optimize_images_task(
    extraction_result: Tuple[LinkedKnowledge, str],
) -> LinkedKnowledge:
    """
    Optimize extracted images.

    Args:
        extraction_result: Tuple of (LinkedKnowledge, image_output_dir)

    Returns:
        LinkedKnowledge object
    """
    from blockether_catalyst.knowledge.optimization.ImageOptimizer import ImageOptimizer

    linked_knowledge, image_output_dir = extraction_result

    logger.info("[Stage 12] Starting image optimization in %s", image_output_dir)

    image_dir = Path(image_output_dir)
    if image_dir.exists():
        # Check if there are any PNG images to optimize
        png_files = list(image_dir.glob("*.png"))
        if png_files:
            optimizer = ImageOptimizer(directory=image_dir, level=3, max_workers=4)
            optimizer.optimize()
            logger.info("[Stage 12] Optimized %d images", len(png_files))
        else:
            logger.info("[Stage 12] No PNG images found to optimize")
    else:
        logger.info("[Stage 12] Image directory does not exist, skipping optimization")

    return linked_knowledge


# ============================================================================
# ORCHESTRATION WORKFLOW
# ============================================================================


def _set_terms_progress(
    workflow_task_id: Optional[str],
    current: int,
    total: int,
    status: str,
) -> None:
    """Update the orchestrator task's progress metadata safely."""
    if not workflow_task_id:
        return

    meta = {"current": current, "total": total, "status": status}
    try:
        celery_app.backend.store_result(
            workflow_task_id,
            result=None,
            state="PROGRESS",
            request=None,
            meta=meta,
        )
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to update progress for workflow %s", workflow_task_id)


@celery_app.task(name="orchestrate.track_terms_stage")
def _track_terms_stage(
    stage_results: Sequence[Any],
    workflow_task_id: Optional[str],
    current: int,
    total: int,
    status: str,
) -> Any:
    """Update progress metadata and unwrap the single result produced by a chord."""
    _set_terms_progress(workflow_task_id, current, total, status)

    if isinstance(stage_results, list) and len(stage_results) == 1:
        return stage_results[0]
    return stage_results


@celery_app.task(name="orchestrate.finalize_terms_workflow")
def _finalize_terms_workflow(
    stage_results: Sequence[Any],
    workflow_task_id: Optional[str],
    start_time: float,
) -> Any:
    """Mark the workflow as completed and return the final knowledge graph."""
    _set_terms_progress(workflow_task_id, 10, 10, "Building knowledge graph")

    final_result = (
        stage_results[0]
        if isinstance(stage_results, list) and stage_results
        else stage_results
    )
    elapsed = time.time() - start_time
    logger.info("[Terms Workflow] All stages completed successfully in %.2fs", elapsed)
    return final_result


@celery_app.task(name="orchestrate.process_terms", bind=True)
def process_terms_workflow(
    self,
    classified_chunks: List[KnowledgeExtractionResultWithChunks],
    model_settings: ExtractionModelSettings,
    knowledge_settings: KnowledgeProcessorSettings,
    extraction_timestamp: int,
    output_dir: str,
    original_pdf_files: List[str],
) -> LinkedKnowledge:
    """Process terms extraction, enrichment, and linking workflow using chords with direct execution."""
    try:
        start_time = time.time()
        workflow_task_id = getattr(self.request, "id", None)
        total_stages = 10

        # Build document to chunks index (needed for co-occurrence and meaning enrichment)
        document_to_chunks_index: Dict[str, List[RawKnowledgeChunk]] = {}
        for result in classified_chunks:
            document_to_chunks_index[result.document_filename] = result.chunks

        logger.info("[Terms Workflow] Starting individual term processing stages")

        # Provide an initial progress update
        _set_terms_progress(
            workflow_task_id,
            current=4,
            total=total_stages,
            status="Preparing term workflow",
        )

        # Build the complete pipeline
        pipeline = chain(
            chord(
                [
                    extract_terms_task.s(classified_chunks, model_settings),
                ],
                _track_terms_stage.s(
                    workflow_task_id,
                    5,
                    total_stages,
                    "Extracting terms",
                ),
            ),
            chord(
                [
                    group_terms_task.s(model_settings),
                ],
                _track_terms_stage.s(
                    workflow_task_id,
                    6,
                    total_stages,
                    "Grouping terms",
                ),
            ),
            chord(
                [
                    enrich_cooccurrences_task.s(
                        document_to_chunks_index,
                        model_settings,
                    ),
                ],
                _track_terms_stage.s(
                    workflow_task_id,
                    7,
                    total_stages,
                    "Analyzing co-occurrences",
                ),
            ),
            chord(
                [
                    enrich_meanings_task.s(
                        document_to_chunks_index,
                        model_settings,
                        knowledge_settings,
                    ),
                ],
                _track_terms_stage.s(
                    workflow_task_id,
                    8,
                    total_stages,
                    "Enriching semantics",
                ),
            ),
            chord(
                [
                    link_terms_task.s(model_settings),
                ],
                _track_terms_stage.s(
                    workflow_task_id,
                    9,
                    total_stages,
                    "Linking terms",
                ),
            ),
            chord(
                [
                    build_linked_knowledge_task.s(
                        classified_chunks,
                        extraction_timestamp,
                        model_settings,
                        output_dir,
                        original_pdf_files,
                    ),
                ],
                _finalize_terms_workflow.s(
                    workflow_task_id,
                    start_time,
                ),
            ),
        )

        # Execute the pipeline directly and wait for the result
        logger.info(
            "[Terms Workflow] Executing pipeline with task_id: %s_pipeline",
            workflow_task_id,
        )
        async_result = pipeline.apply_async(
            task_id=f"{workflow_task_id}_pipeline" if workflow_task_id else None
        )

        # Wait for the pipeline to complete and return the result
        # Use disable_sync_subtasks to prevent the PROGRESS state issue
        result = async_result.get(disable_sync_subtasks=False, timeout=300)

        logger.info("[Terms Workflow] Pipeline completed successfully")
        return result

    except Exception:
        logger.exception("Terms workflow failed")
        raise  # Re-raise with original traceback  # Re-raise with original traceback
