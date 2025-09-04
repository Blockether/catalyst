"""
Hierarchical Knowledge Extraction Core.

This module processes documents hierarchically by analyzing them in batches
to build a knowledge tree structure with terms and rationales.
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from com_blockether_catalyst.knowledge.internal import PDFKnowledgeExtractor
from com_blockether_catalyst.knowledge.internal.HierarchicalExtractionCallBase import (
    BaseHierarchicalExtractionCall,
)
from com_blockether_catalyst.knowledge.internal.HierarchicalExtractionTypes import (
    CumulativeKnowledgeState,
    DocumentAnalysisResult,
    ExtractedTerm,
    HierarchicalExtractionResponse,
    KnowledgeNode,
    PageBatchAnalysis,
)
from com_blockether_catalyst.knowledge.internal.KnowledgeExtractionTypes import (
    KnowledgeExtractionItem,
    KnowledgeExtractionOutput,
    KnowledgeExtractionResult,
    KnowledgePageData,
)

logger = logging.getLogger(__name__)


class HierarchicalKnowledgeExtractionCore:
    """Core for hierarchical knowledge extraction from documents."""

    def __init__(
        self,
        extraction_call: BaseHierarchicalExtractionCall,
        batch_size: int = 4,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the hierarchical extraction core.

        Args:
            extraction_call: Call for hierarchical extraction
            batch_size: Number of pages per batch (default 4)
            output_dir: Optional output directory for results
        """
        self._extraction_call = extraction_call
        self._batch_size = batch_size
        self._output_dir = output_dir or Path("hierarchical_extraction")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize PDF extractor
        self._pdf_extractor = PDFKnowledgeExtractor()

        # Define extractors for each supported extension
        self._extractors = {".pdf": self._pdf_extractor}

        logger.info(f"HierarchicalKnowledgeExtractionCore initialized with batch_size={batch_size}")

    def _resolve_glob_patterns(self, globs: List[str]) -> List[Path]:
        """
        Resolve glob patterns to actual file paths.

        Args:
            globs: List of glob patterns or file paths

        Returns:
            List of resolved file paths
        """
        all_files = []
        for glob_pattern in globs:
            path = Path(glob_pattern)
            if path.is_file():
                # Direct file path
                all_files.append(path)
            else:
                # Glob pattern
                if "*" in glob_pattern or "?" in glob_pattern or "[" in glob_pattern:
                    # It's a glob pattern
                    parent = Path(glob_pattern).parent if "/" in glob_pattern else Path(".")
                    pattern = Path(glob_pattern).name
                    matches = list(parent.glob(pattern))
                    all_files.extend(matches)
                else:
                    # It might be a directory or non-existent path
                    path_obj = Path(glob_pattern)
                    if path_obj.is_dir():
                        # Get all files in directory
                        all_files.extend(path_obj.iterdir())
        return all_files

    def _group_files_by_extension(self, files: List[Path]) -> Dict[str, List[Path]]:
        """
        Group files by their extension.

        Args:
            files: List of file paths

        Returns:
            Dictionary mapping extension to list of file paths
        """
        files_by_extension = defaultdict(list)
        for file_path in files:
            if file_path.is_file():
                extension = file_path.suffix.lower()
                files_by_extension[extension].append(file_path)
        return files_by_extension

    async def extract(self, globs: List[str]) -> List[DocumentAnalysisResult]:
        """
        Extract hierarchical knowledge from files matching glob patterns.

        Args:
            globs: List of glob patterns to match files

        Returns:
            List of DocumentAnalysisResult for each processed document
        """
        logger.info(f"Starting hierarchical extraction with {len(globs)} glob patterns")

        # Resolve all glob patterns to actual file paths
        all_files = self._resolve_glob_patterns(globs)
        logger.info(f"Found {len(all_files)} files from glob patterns")

        # Group files by extension
        files_by_extension = self._group_files_by_extension(all_files)
        logger.info(f"Grouped files by extension: {dict((k, len(v)) for k, v in files_by_extension.items())}")

        # Process all documents and collect results
        all_results = []

        for extension, file_list in files_by_extension.items():
            if extension not in self._extractors:
                logger.warning(f"Skipping unsupported extension: {extension}")
                continue

            logger.info(f"Processing {len(file_list)} {extension} files")

            for file_path in file_list:
                logger.info(f"Processing: {file_path}")
                result = await self._analyze_document(file_path)
                if result:
                    all_results.append(result)

        logger.info(f"Completed hierarchical extraction: {len(all_results)} documents processed")
        return all_results

    async def _analyze_document(self, file_path: Path) -> Optional[DocumentAnalysisResult]:
        """
        Analyze a single document hierarchically.

        Args:
            file_path: Path to document

        Returns:
            DocumentAnalysisResult or None if extraction failed
        """
        logger.info(f"Starting hierarchical analysis of: {file_path.name}")

        # Extract raw document based on extension
        extension = file_path.suffix.lower()
        if extension not in self._extractors:
            logger.error(f"Unsupported file type: {extension}")
            return None

        extractor = self._extractors[extension]
        raw_result = extractor.extract(file_path)

        if not raw_result:
            logger.error(f"Failed to extract: {file_path}")
            return None

        pages = raw_result.pages
        document_name = file_path.name

        # Initialize cumulative state
        cumulative_state = CumulativeKnowledgeState(
            document_name=document_name,
            knowledge_tree=[],
            all_keywords={},
            all_acronyms={},
        )

        # Process in batches
        total_batches = (len(pages) + self._batch_size - 1) // self._batch_size
        batch_analyses = []

        for batch_num in range(total_batches):
            start_idx = batch_num * self._batch_size
            end_idx = min(start_idx + self._batch_size, len(pages))
            batch_pages = pages[start_idx:end_idx]

            logger.info(f"Processing batch {batch_num + 1}/{total_batches} " f"(pages {start_idx + 1}-{end_idx})")

            # Extract hierarchical knowledge from batch
            result = await self._extraction_call.execute(
                pages=batch_pages,
                document_name=document_name,
                cumulative_state=cumulative_state,
                batch_number=batch_num + 1,
                total_batches_estimate=total_batches,
            )

            # Update cumulative state
            batch_analysis = result.final_response.batch_analysis
            batch_analyses.append(batch_analysis)

            cumulative_state = self._update_cumulative_state(
                cumulative_state,
                result.final_response,
                batch_analysis,
            )

            logger.debug(
                f"Batch {batch_num + 1}: Found {len(batch_analysis.sections_found)} sections, "
                f"{len(batch_analysis.keywords)} keywords, {len(batch_analysis.acronyms)} acronyms"
            )

        # Build final result
        result = self._build_final_result(
            cumulative_state,
            batch_analyses,
            len(pages),
        )

        logger.info(
            f"Analysis complete: {result.total_sections} sections, "
            f"{result.total_keywords} keywords, {result.total_acronyms} acronyms "
            f"in {result.total_pages} pages"
        )

        return result

    def _update_cumulative_state(
        self,
        state: CumulativeKnowledgeState,
        extraction_response: HierarchicalExtractionResponse,
        batch_analysis: PageBatchAnalysis,
    ) -> CumulativeKnowledgeState:
        """
        Update cumulative state with batch results.

        Args:
            state: Current cumulative state
            extraction_response: Response from extraction
            batch_analysis: Analysis of current batch

        Returns:
            Updated cumulative state
        """
        # Update basic counters
        state.pages_processed += batch_analysis.page_end - batch_analysis.page_start + 1
        state.batches_processed += 1

        # Add new sections to hierarchy
        for new_section in extraction_response.new_sections:
            self._add_section_to_hierarchy(state.knowledge_tree, new_section)

        # Update existing sections
        for updated_section in extraction_response.updated_sections:
            self._update_section_in_hierarchy(state.knowledge_tree, updated_section)

        # Add keywords with rationales
        for keyword in batch_analysis.keywords:
            if keyword.term not in state.all_keywords:
                state.all_keywords[keyword.term] = keyword
            else:
                # Merge keyword information
                existing = state.all_keywords[keyword.term]
                if keyword.definition and not existing.definition:
                    existing.definition = keyword.definition
                if keyword.rationale:
                    existing.rationale = keyword.rationale
                existing.relationships.extend(keyword.relationships)
                existing.relationships = list(set(existing.relationships))
                existing.importance = max(existing.importance, keyword.importance)

        # Add acronyms with rationales
        for acronym in batch_analysis.acronyms:
            if acronym.term not in state.all_acronyms:
                state.all_acronyms[acronym.term] = acronym
            else:
                # Merge acronym information
                existing = state.all_acronyms[acronym.term]
                if acronym.full_form and not existing.full_form:
                    existing.full_form = acronym.full_form
                if acronym.definition and not existing.definition:
                    existing.definition = acronym.definition
                if acronym.rationale:
                    existing.rationale = acronym.rationale
                existing.relationships.extend(acronym.relationships)
                existing.relationships = list(set(existing.relationships))
                existing.importance = max(existing.importance, acronym.importance)

        # Update context
        state.last_batch_summary = batch_analysis.batch_summary

        # Update document type if identified
        if extraction_response.document_type_assessment:
            state.document_type = extraction_response.document_type_assessment

        return state

    def _add_section_to_hierarchy(
        self,
        tree: List[KnowledgeNode],
        new_section: KnowledgeNode,
    ):
        """
        Add a new section to the knowledge hierarchy.

        Args:
            tree: Current knowledge tree
            new_section: Section to add
        """
        # Find appropriate parent based on level
        if new_section.level == 1:
            # Top-level section
            tree.append(new_section)
        else:
            # Find parent section
            parent = self._find_parent_for_level(tree, new_section.level)
            if parent:
                parent.children.append(new_section)
            else:
                # If no appropriate parent, add as top-level
                tree.append(new_section)

    def _update_section_in_hierarchy(
        self,
        tree: List[KnowledgeNode],
        updated_section: KnowledgeNode,
    ):
        """
        Update an existing section in the hierarchy.

        Args:
            tree: Knowledge tree
            updated_section: Section with updates
        """
        # Find and update the section
        existing = self._find_section_by_title(tree, updated_section.title)
        if existing:
            # Update fields
            existing.summary = updated_section.summary or existing.summary
            existing.page_end = max(existing.page_end, updated_section.page_end)
            existing.key_concepts.extend(updated_section.key_concepts)
            existing.key_concepts = list(set(existing.key_concepts))

    def _find_parent_for_level(
        self,
        tree: List[KnowledgeNode],
        target_level: int,
    ) -> Optional[KnowledgeNode]:
        """
        Find appropriate parent node for a given level.

        Args:
            tree: Knowledge tree
            target_level: Level of node to insert

        Returns:
            Parent node or None
        """
        # Look for most recent node at level - 1
        for node in reversed(tree):
            if node.level == target_level - 1:
                return node
            # Recursively check children
            parent = self._find_parent_for_level(node.children, target_level)
            if parent:
                return parent
        return None

    def _find_section_by_title(
        self,
        tree: List[KnowledgeNode],
        title: str,
    ) -> Optional[KnowledgeNode]:
        """
        Find a section by title.

        Args:
            tree: Knowledge tree
            title: Section title to find

        Returns:
            Found node or None
        """
        for node in tree:
            if node.title == title:
                return node
            # Check children
            found = self._find_section_by_title(node.children, title)
            if found:
                return found
        return None

    def _build_final_result(
        self,
        state: CumulativeKnowledgeState,
        batch_analyses: List[PageBatchAnalysis],
        total_pages: int,
    ) -> DocumentAnalysisResult:
        """
        Build the final analysis result.

        Args:
            state: Final cumulative state
            batch_analyses: All batch analyses
            total_pages: Total pages in document

        Returns:
            Complete document analysis result
        """
        # Build executive summary
        executive_summary = self._build_executive_summary(state, batch_analyses)

        # Build detailed summary
        detailed_summary = self._build_detailed_summary(state, batch_analyses)

        # Build table of contents
        toc = self._build_table_of_contents(state.knowledge_tree)

        # Count sections
        total_sections = self._count_sections(state.knowledge_tree)

        # Assess complexity
        complexity = self._assess_overall_complexity(batch_analyses)

        # Convert dictionaries to lists for the result
        keywords_list = list(state.all_keywords.values())
        acronyms_list = list(state.all_acronyms.values())

        return DocumentAnalysisResult(
            document_name=state.document_name,
            document_type=state.document_type or "unknown",
            knowledge_tree=state.knowledge_tree,
            keywords=keywords_list,
            acronyms=acronyms_list,
            executive_summary=executive_summary,
            detailed_summary=detailed_summary,
            total_pages=total_pages,
            total_sections=total_sections,
            total_keywords=len(keywords_list),
            total_acronyms=len(acronyms_list),
            complexity_assessment=complexity,
            table_of_contents=toc,
        )

    def _build_executive_summary(
        self,
        state: CumulativeKnowledgeState,
        batch_analyses: List[PageBatchAnalysis],
    ) -> str:
        """Build executive summary from accumulated knowledge."""
        # Combine key points from all batches
        key_points = []
        for analysis in batch_analyses[:5]:  # Focus on early batches
            if analysis.content_type in ["introduction", "overview"]:
                key_points.append(analysis.batch_summary)

        if state.overall_summary:
            return state.overall_summary

        return " ".join(key_points) if key_points else "Document analysis summary pending."

    def _build_detailed_summary(
        self,
        state: CumulativeKnowledgeState,
        batch_analyses: List[PageBatchAnalysis],
    ) -> str:
        """Build detailed summary."""
        summaries = [analysis.batch_summary for analysis in batch_analyses]
        return "\n\n".join(summaries)

    def _build_table_of_contents(
        self,
        tree: List[KnowledgeNode],
        level: int = 0,
    ) -> List[Dict[str, Any]]:
        """Build flattened table of contents."""
        toc = []
        for node in tree:
            toc.append(
                {
                    "title": node.title,
                    "level": node.level,
                    "page_start": node.page_start,
                    "page_end": node.page_end,
                    "indent": "  " * level,
                }
            )
            # Add children
            toc.extend(self._build_table_of_contents(node.children, level + 1))
        return toc

    def _count_sections(self, tree: List[KnowledgeNode]) -> int:
        """Count total sections in tree."""
        count = len(tree)
        for node in tree:
            count += self._count_sections(node.children)
        return count

    def _assess_overall_complexity(
        self,
        batch_analyses: List[PageBatchAnalysis],
    ) -> str:
        """Assess overall document complexity."""
        complexities = [a.complexity_level for a in batch_analyses]
        # Return most common complexity
        from collections import Counter

        most_common = Counter(complexities).most_common(1)
        return most_common[0][0] if most_common else "medium"
