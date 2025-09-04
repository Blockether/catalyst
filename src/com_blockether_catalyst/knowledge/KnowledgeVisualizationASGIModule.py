"""
Knowledge Visualization ASGI Module - FastAPI + HTMX + Tailwind CSS visualization for LinkedKnowledge
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import uvicorn
from fastapi import APIRouter, FastAPI, Form, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import Field

from com_blockether_catalyst.asgi.ASGICoreModule import ASGICoreModule
from com_blockether_catalyst.knowledge.internal.KnowledgeExtractionTypes import (
    KnowledgeChunkWithTerms,
)

from .internal.KnowledgeExtractionTypes import (
    LinkedKnowledge,
    Term,
)
from .KnowledgeSearchCore import (
    KnowledgeSearchCore,
    KnowledgeSearchResult,
    SimilaritySearchResult,
)

logger = logging.getLogger(__name__)


class KnowledgeVisualizationASGIModule(ASGICoreModule):
    """Web-based visualization for LinkedKnowledge using FastAPI + HTMX + Tailwind."""

    # Add Pydantic fields for this module
    output_dir: Path = Field(
        default_factory=lambda: Path("public/knowledge_extraction"),
        description="Directory containing knowledge extraction outputs",
    )
    linked_knowledge: Optional[LinkedKnowledge] = Field(
        default=None, exclude=True, description="Loaded LinkedKnowledge data"
    )
    search_core: Optional[KnowledgeSearchCore] = Field(default=None, exclude=True, description="Knowledge search core")

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        prefix: str = "/knowledge",
        **kwargs: Any,
    ) -> None:
        """Initialize the web visualization module.

        Args:
            output_dir: Directory containing knowledge extraction outputs
            prefix: URL prefix for this module
            **kwargs: Additional arguments passed to parent
        """
        # Prepare initialization data
        init_data = {
            "prefix": prefix,
            "title": "Knowledge Visualization",
            "description": "Web-based visualization for LinkedKnowledge",
            "template_dirs": [Path(__file__).parent / "templates"],
            "htmx_enabled": True,
            **kwargs,
        }

        if output_dir is not None:
            init_data["output_dir"] = output_dir

        # Initialize parent ASGICoreModule with all fields
        super().__init__(**init_data)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def app(self) -> FastAPI:
        """Create a FastAPI app instance with this module's routes.

        This is primarily for testing purposes. In production, the module
        should be mounted to an ASGICoreApplication.

        Returns:
            FastAPI app with module routes configured
        """
        if not hasattr(self, "_app"):
            self._app = FastAPI(title=self.title or "Knowledge Visualization")
            router = APIRouter()
            self.setup_routes(router)
            # Include router with prefix
            self._app.include_router(router, prefix=self.prefix)
        return self._app

    def load_from_pickle(self, pickle_path: Path, create_search_core: bool = True) -> None:
        """Load LinkedKnowledge from pickle file.

        Args:
            pickle_path: Path to the pickle file
            create_search_core: If True, create a new KnowledgeSearchCore.
                              Set to False if you'll provide your own search_core.

        Returns:
            LinkedKnowledge object
        """
        with open(pickle_path, "rb") as f:
            self.linked_knowledge = pickle.load(f)

        # Initialize search core with loaded knowledge only if requested
        if self.linked_knowledge and create_search_core:
            self.search_core = KnowledgeSearchCore(self.linked_knowledge)

        return None

    def _get_document_filename(self, doc_id: str) -> Optional[str]:
        """Get the filename for a document if it exists in source_documents.

        Args:
            doc_id: Document ID

        Returns:
            Filename if found, None otherwise
        """
        docs_dir = self.output_dir / "source_documents"
        if not docs_dir.exists():
            return None

        # Get document name from linked knowledge
        if self.linked_knowledge and doc_id in self.linked_knowledge.documents:
            doc = self.linked_knowledge.documents[doc_id]
            if hasattr(doc, "filename"):
                # Look for matching file in source_documents
                for file_path in docs_dir.iterdir():
                    if doc.filename in file_path.name:
                        return file_path.name
        return None

    def setup_routes(self, router: APIRouter) -> None:
        """Set up FastAPI routes."""

        @router.get("/", response_class=HTMLResponse)
        async def index(request: Request) -> str:
            """Main page with HTMX and Tailwind."""
            return self._render_main_page()

        @router.get("/api/stats")
        async def get_stats() -> dict[str, int | str]:
            """Get statistics for the dashboard."""
            if not self.linked_knowledge:
                return {"error": "No data loaded"}

            return {
                "total_documents": len(self.linked_knowledge.documents),
                "total_terms": len(self.linked_knowledge.terms),
                "total_acronyms": self.linked_knowledge.total_acronyms,
                "total_keywords": self.linked_knowledge.total_keywords,
                "total_links": len(self.linked_knowledge.links),
                "total_chunks": self.linked_knowledge.total_chunks,
            }

        @router.get("/api/documents", response_class=HTMLResponse)
        async def get_documents(page: int = Query(1, ge=1), per_page: int = Query(10, ge=1, le=50)) -> str:
            """Get paginated documents list as HTML partial for HTMX."""
            if not self.linked_knowledge:
                return "<div>No data loaded</div>"

            # Get all documents and paginate
            all_docs = list(self.linked_knowledge.documents.items())
            total = len(all_docs)
            start = (page - 1) * per_page
            end = start + per_page
            docs_page = all_docs[start:end]
            total_pages = (total + per_page - 1) // per_page

            html = []

            # Add pagination info at the top
            html.append(
                f'<div class="text-sm text-gray-600 mb-3">Showing {start + 1}-{min(end, total)} of {total} documents</div>'
            )

            for doc_id, doc in docs_page:
                # Get the document filename for the direct link
                doc_filename = self._get_document_filename(doc_id)
                doc_link_html = ""
                if doc_filename:
                    doc_url = f"/documents/{doc_filename}"
                    doc_link_html = f"""
                        <a href="{doc_url}" target="_blank"
                           class="text-lg font-semibold mb-2 text-blue-600 hover:text-blue-800 hover:underline"
                           onclick="event.stopPropagation();">
                            {doc.filename if hasattr(doc, "filename") else "Unknown"}
                        </a>
                    """
                else:
                    doc_link_html = f'<h3 class="text-lg font-semibold mb-2">{doc.filename if hasattr(doc, "filename") else "Unknown"}</h3>'

                html.append(
                    f"""
                <div class="bg-white rounded-lg shadow p-6 mb-3 hover:shadow-xl transition-all cursor-pointer"
                     hx-get="api/documents/{doc_id}"
                     hx-target="#document-details"
                     hx-swap="innerHTML">
                    <div class="flex justify-between items-start">
                        <div class="flex-1">
                            {doc_link_html}
                            <p class="text-xs text-gray-500 font-mono mb-3">ID: {doc_id[:32]}...</p>
                            <div class="flex gap-4 text-sm text-gray-600">
                                <span class="flex items-center gap-1">
                                    <span class="font-semibold">{doc.total_pages}</span>
                                    <span class="text-gray-400">pages</span>
                                </span>
                                <span class="flex items-center gap-1">
                                    <span class="font-semibold">{doc.total_chunks}</span>
                                    <span class="text-gray-400">chunks</span>
                                </span>
                                <span class="flex items-center gap-1">
                                    <span class="font-semibold">{doc.total_tables if hasattr(doc, "total_tables") else 0}</span>
                                    <span class="text-gray-400">tables</span>
                                </span>
                            </div>
                        </div>
                        <div class="text-gray-400">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                            </svg>
                        </div>
                    </div>
                </div>
                """
                )

            # Add pagination controls
            if total_pages > 1:
                html.append('<div class="flex justify-center items-center gap-2 mt-4">')

                # Previous button
                if page > 1:
                    html.append(
                        f"""
                        <button class="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300"
                                hx-get="api/documents?page={page - 1}&per_page={per_page}"
                                hx-target="#documents-list"
                                hx-swap="innerHTML">
                            ← Previous
                        </button>
                    """
                    )

                # Page numbers
                for p in range(max(1, page - 2), min(total_pages + 1, page + 3)):
                    if p == page:
                        html.append(f'<span class="px-3 py-1 text-sm bg-yellow-400 rounded font-semibold">{p}</span>')
                    else:
                        html.append(
                            f"""
                            <button class="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300"
                                    hx-get="api/documents?page={p}&per_page={per_page}"
                                    hx-target="#documents-list"
                                    hx-swap="innerHTML">
                                {p}
                            </button>
                        """
                        )

                # Next button
                if page < total_pages:
                    html.append(
                        f"""
                        <button class="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300"
                                hx-get="api/documents?page={page + 1}&per_page={per_page}"
                                hx-target="#documents-list"
                                hx-swap="innerHTML">
                            Next →
                        </button>
                    """
                    )

                html.append("</div>")

            return "".join(html)

        @router.get("/api/documents/{doc_id}/chunks", response_class=HTMLResponse)
        async def get_document_chunks(doc_id: str) -> str:
            """Get chunks view for a document."""
            if not self.linked_knowledge or doc_id not in self.linked_knowledge.documents:
                return "<div class='text-red-500'>Document not found</div>"

            chunk_ids = self.linked_knowledge.document_to_chunk_ids_index.get(doc_id, set())
            # Sort chunks by their index to display in ascending order
            sorted_chunk_ids = sorted(
                chunk_ids,
                key=lambda cid: self.linked_knowledge.chunks[cid].index if self.linked_knowledge else 0,
            )
            html = []
            html.append('<div class="space-y-3 max-h-[720px] overflow-y-auto">')

            # Add chunks with their associated terms
            for i, chunk_id in enumerate(sorted_chunk_ids):
                chunk = self.linked_knowledge.chunks[chunk_id]
                chunk_index = chunk.index
                # Create both truncated and full text versions
                is_truncated = len(chunk.text) > 500
                chunk_text_short = chunk.text[:500] + "..." if is_truncated else chunk.text
                chunk_text_full = chunk.text
                # Don't escape HTML - let the markdown renderer handle it

                # Find terms in this chunk
                chunk_acronyms: list[str] = []
                chunk_keywords: list[str] = []

                for term_id, term in self.linked_knowledge.terms.items():
                    if hasattr(term, "occurrences"):
                        for occ in term.occurrences:
                            if occ.document_id == doc_id and occ.chunk_index == chunk_index:
                                if term.term_type == "acronym":
                                    chunk_acronyms.append(term.term)
                                elif term.term_type == "keyword":
                                    chunk_keywords.append(term.term)
                                break

                html.append(
                    f"""
                    <div class="bg-gray-50 rounded-lg p-4 border border-gray-200 hover:border-gray-300 transition-colors">
                        <div class="flex justify-between items-start mb-3 gap-4">
                            <div class="flex flex-wrap gap-1 items-center" style="max-width: calc(100% - 168px);">
                                <div id="chunk-terms-{
                        chunk_index
                    }" class="flex flex-wrap gap-1 items-center">
                                    {
                        "".join(
                            [
                                f'<span class="inline-block px-2 py-1 text-xs rounded font-medium" style="background-color: #fbbf24; color: #000;">📝 {acronym}</span>'
                                for acronym in chunk_acronyms[:5]
                            ]
                        )
                        if chunk_acronyms
                        else ""
                    }
                                    {
                        "".join(
                            [
                                f'<span class="inline-block px-2 py-1 text-xs rounded font-medium" style="background-color: #1e40af; color: #fff;">🔑 {keyword}</span>'
                                for keyword in chunk_keywords[:5]
                            ]
                        )
                        if chunk_keywords
                        else ""
                    }
                                </div>
                                <div id="chunk-all-terms-{
                        chunk_index
                    }" class="hidden flex-wrap gap-1 items-center">
                                    {
                        "".join(
                            [
                                f'<span class="inline-block px-2 py-1 text-xs rounded font-medium" style="background-color: #fbbf24; color: #000;">📝 {acronym}</span>'
                                for acronym in chunk_acronyms
                            ]
                        )
                        if chunk_acronyms
                        else ""
                    }
                                    {
                        "".join(
                            [
                                f'<span class="inline-block px-2 py-1 text-xs rounded font-medium" style="background-color: #1e40af; color: #fff;">🔑 {keyword}</span>'
                                for keyword in chunk_keywords
                            ]
                        )
                        if chunk_keywords
                        else ""
                    }
                                </div>
                                {
                        f'''<button onclick="toggleChunkTerms({chunk_index})"
                                     id="chunk-toggle-{chunk_index}"
                                     class="px-2 py-1 text-xs text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200 rounded cursor-pointer transition-colors">
                                     +{len(chunk_acronyms) - 5 + len(chunk_keywords) - 5 if len(chunk_keywords) > 5 else len(chunk_acronyms) - 5} more
                                </button>'''
                        if len(chunk_acronyms) > 5 or len(chunk_keywords) > 5
                        else ""
                    }
                            </div>
                            <div class="flex gap-2 flex-shrink-0">
                                {
                        self._render_page_link(
                            doc_id, chunk.page, f"📄 Page {chunk.page}"
                        )
                        if hasattr(chunk, "page")
                        else "<span class='px-2 py-1 bg-white rounded text-xs'>📄 Page Unknown</span>"
                    }
                                <span class="px-2 py-1 bg-white rounded text-xs text-gray-500">📦 Chunk {
                        chunk_index + 1
                    }</span>
                                {
                        self._render_pdf_link(doc_id, chunk.page)
                        if hasattr(chunk, "page")
                        else ""
                    }
                            </div>
                        </div>

                        <div class="text-sm text-gray-700 leading-relaxed pt-3">
                            <div id="chunk-text-short-{chunk_index}" class="{
                        "hidden" if False else ""
                    }" data-markdown="{
                        chunk_text_short.replace('"', "&quot;").replace("\n", "\\n")
                    }">
                                {chunk_text_short}
                            </div>
                            <div id="chunk-text-full-{
                        chunk_index
                    }" class="hidden" data-markdown="{
                        chunk_text_full.replace('"', "&quot;").replace("\n", "\\n")
                    }">
                                {chunk_text_full}
                            </div>
                            {
                        f'''<button onclick="toggleChunkText({chunk_index})"
                                     id="chunk-text-toggle-{chunk_index}"
                                     class="mt-2 px-3 py-1 text-sm text-blue-600 hover:text-blue-800 bg-blue-50 hover:bg-blue-100 rounded cursor-pointer transition-colors">
                                     Expand
                                </button>'''
                        if is_truncated
                        else ""
                    }
                        </div>

                        <div class="flex justify-between items-center mt-3 pt-3 border-t border-gray-200">
                            <div class="text-xs text-gray-500">{
                        len(chunk.text)
                    } characters</div>
                            <div class="text-xs text-gray-500">
                                {len(chunk_acronyms)} acronyms, {
                        len(chunk_keywords)
                    } keywords
                            </div>
                        </div>
                    </div>
                """
                )

            html.append("</div>")
            return "".join(html)

        @router.get("/api/documents/{doc_id}/tables", response_class=HTMLResponse)
        async def get_document_tables(doc_id: str) -> str:
            """Get tables view for a document."""
            if not self.linked_knowledge or doc_id not in self.linked_knowledge.documents:
                return "<div class='text-red-500'>Document not found</div>"

            pages = self.linked_knowledge.pages
            # Get tables from pages with their page numbers
            tables = []

            for page_num in range(len(pages)):
                page_data = pages.get((doc_id, page_num))
                if page_data:
                    for table in page_data.tables:
                        tables.append((page_num, table))

            html = [f'<h3 class="text-lg font-semibold mb-3">Document Tables ({len(tables)} total)</h3>']
            html.append('<div class="space-y-4 max-h-[720px] overflow-y-hidden">')

            if tables:
                for page_num, table in tables:
                    html.append(
                        f"""
                        <div class="bg-white border border-gray-200 rounded-lg p-4">
                            <div class="mb-2 text-sm font-semibold text-gray-700">
                                📊 Table on {self._render_page_link(doc_id, page_num, f"Page {page_num}") if self._get_document_filename(doc_id) else f"Page {page_num}"}
                            </div>
                            <div class="overflow-x-auto">
                                {table.to_html_table()}
                            </div>
                        </div>
                    """
                    )
            else:
                html.append('<div class="text-gray-500 text-center py-8">No tables found in this document</div>')

            html.append("</div>")
            return "".join(html)

        @router.get("/api/documents/{doc_id}/acronyms", response_class=HTMLResponse)
        async def get_document_acronyms(doc_id: str) -> str:
            """Get acronyms view for a document."""
            if not self.linked_knowledge or doc_id not in self.linked_knowledge.documents:
                return "<div class='text-red-500'>Document not found</div>"

            # Find acronyms that appear in this document
            doc_acronyms = []
            for term_id, term in self.linked_knowledge.terms.items():
                if term.term_type == "acronym":
                    if any(occ.document_id == doc_id for occ in term.occurrences):
                        doc_acronyms.append((term_id, term))

            # Sort by frequency in this document
            doc_acronyms.sort(
                key=lambda item: sum(1 for o in item[1].occurrences if o.document_id == doc_id),
                reverse=True,
            )

            html = [f'<h3 class="text-lg font-semibold mb-3">Document Acronyms ({len(doc_acronyms)} unique)</h3>']
            html.append('<div class="space-y-3 max-h-[720px] overflow-y-auto">')

            if doc_acronyms:
                for term_id, acronym in doc_acronyms:
                    doc_occurrences = [o for o in acronym.occurrences if o.document_id == doc_id]
                    html.append(
                        f"""
                        <div class="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                             hx-get="api/term/{term_id}"
                             hx-target="#modalContent"
                             hx-trigger="click"
                             hx-on::after-request="document.getElementById('termModal').classList.remove('hidden')">
                            <div class="flex items-start justify-between">
                                <div class="flex-1">
                                    <div class="flex items-center gap-2 mb-2">
                                        <span class="px-2 py-1 text-xs font-semibold rounded" style="background-color: #fbbf24; color: #000;">
                                            📝 Acronym
                                        </span>
                                        <span class="text-lg font-semibold">{acronym.term}</span>
                                    </div>
                                    {f'<p class="text-sm font-medium text-gray-700">Full form: {acronym.full_form}</p>'}
                                    <p class="text-sm text-gray-600 mt-1">{acronym.meaning or "No description"}</p>
                                    <div class="flex gap-4 mt-2 text-xs text-gray-500">
                                        <span>Appears {len(doc_occurrences)} times in this document</span>
                                        <span>Pages: {", ".join(str(o.page) for o in doc_occurrences[:5])}{"+" + str(len(doc_occurrences) - 5) + " more" if len(doc_occurrences) > 5 else ""}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    """
                    )
            else:
                html.append('<div class="text-gray-500 text-center py-8">No acronyms found in this document</div>')

            html.append("</div>")
            return "".join(html)

        @router.get("/api/documents/{doc_id}/keywords", response_class=HTMLResponse)
        async def get_document_keywords(doc_id: str) -> str:
            """Get keywords view for a document."""
            if not self.linked_knowledge or doc_id not in self.linked_knowledge.documents:
                return "<div class='text-red-500'>Document not found</div>"

            # Find keywords that appear in this document
            doc_keywords = []
            for term_id, term in self.linked_knowledge.terms.items():
                if term.term_type == "keyword":
                    if any(occ.document_id == doc_id for occ in term.occurrences):
                        doc_keywords.append((term_id, term))

            # Sort by frequency in this document
            doc_keywords.sort(
                key=lambda item: sum(1 for o in item[1].occurrences if o.document_id == doc_id),
                reverse=True,
            )

            html = [f'<h3 class="text-lg font-semibold mb-3">Document Keywords ({len(doc_keywords)} unique)</h3>']
            html.append('<div class="space-y-3 max-h-[720px] overflow-y-auto">')

            if doc_keywords:
                for term_id, keyword in doc_keywords:
                    doc_occurrences = [o for o in keyword.occurrences if o.document_id == doc_id]
                    html.append(
                        f"""
                        <div class="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                             hx-get="api/term/{term_id}"
                             hx-target="#modalContent"
                             hx-trigger="click"
                             hx-on::after-request="document.getElementById('termModal').classList.remove('hidden')">
                            <div class="flex items-start justify-between">
                                <div class="flex-1">
                                    <div class="flex items-center gap-2 mb-2">
                                        <span class="px-2 py-1 text-xs font-semibold rounded" style="background-color: #1e40af; color: #fff;">
                                            🔑 Keyword
                                        </span>
                                        <span class="text-lg font-semibold">{keyword.term}</span>
                                    </div>
                                    <p class="text-sm text-gray-600 mt-1">{keyword.meaning or "No description"}</p>
                                    <div class="flex gap-4 mt-2 text-xs text-gray-500">
                                        <span>Appears {len(doc_occurrences)} times in this document</span>
                                        <span>Score: {keyword.mean_score:.3f}</span>
                                        <span>Pages: {", ".join(str(o.page) for o in doc_occurrences[:5])}{"+" + str(len(doc_occurrences) - 5) + " more" if len(doc_occurrences) > 5 else ""}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    """
                    )
            else:
                html.append('<div class="text-gray-500 text-center py-8">No keywords found in this document</div>')

            html.append("</div>")
            return "".join(html)

        @router.get("/api/documents/{doc_id}", response_class=HTMLResponse)
        async def get_document_details(doc_id: str) -> str:
            """Get detailed view of a document with its chunks."""
            if not self.linked_knowledge:
                return "<div>No data loaded</div>"

            if doc_id not in self.linked_knowledge.documents:
                return "<div class='text-red-500'>Document not found</div>"

            doc = self.linked_knowledge.documents[doc_id]
            chunk_ids = self.linked_knowledge.document_to_chunk_ids_index.get(doc_id, set())

            html = [
                f"""
            <div class="bg-white rounded-lg shadow-lg p-6">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h2 class="text-2xl font-bold">{doc.filename if hasattr(doc, "filename") else "Unknown"}</h2>
                        <p class="text-sm text-gray-600">Document ID: {doc_id}</p>
                        <div class="flex gap-4 mt-2 text-sm">
                            <span class="text-gray-500">📄 Pages: {doc.total_pages if hasattr(doc, "total_pages") else 0}</span>
                            <span class="text-gray-500">📝 Chunks: {len(chunk_ids)}</span>
                            <span class="text-gray-500">📊 Tables: {doc.total_tables if hasattr(doc, "total_tables") else 0}</span>
                            <span class="text-gray-500">📝 Acronyms: {doc.total_acronyms if hasattr(doc, "total_acronyms") else 0}</span>
                            <span class="text-gray-500">🔑 Keywords: {doc.total_keywords if hasattr(doc, "total_keywords") else 0}</span>
                        </div>
                    </div>
                    <div class="flex gap-2">
                        {self._render_full_document_link(doc_id)}
                        <button class="px-3 py-1 text-gray-600 border border-gray-300 rounded hover:bg-gray-100"
                                hx-get="api/documents"
                                hx-target="#document-details"
                                hx-swap="innerHTML">
                            ← Back
                        </button>
                    </div>
                </div>

                <div class="border-t pt-4" id="document-content-area">
                    <div class="space-y-3 max-h-[720px] overflow-y-auto">
            """
            ]

            # Sort chunks by their index to display in ascending order
            sorted_chunk_ids = sorted(
                chunk_ids,
                key=lambda cid: (
                    self.linked_knowledge.chunks[cid].index
                    if self.linked_knowledge and hasattr(self.linked_knowledge.chunks[cid], "index")
                    else 0
                ),
            )

            # Add chunks with their associated terms
            for i, chunk_id in enumerate(sorted_chunk_ids):
                chunk = self.linked_knowledge.chunks[chunk_id]
                chunk_index = chunk.index if hasattr(chunk, "index") else i
                # Create both truncated and full text versions
                is_truncated = len(chunk.text) > 500
                chunk_text_short = chunk.text[:500] + "..." if is_truncated else chunk.text
                chunk_text_full = chunk.text
                # Don't escape HTML - let the markdown renderer handle it

                # Find terms that appear in this chunk
                chunk_acronyms: list[str] = []
                chunk_keywords: list[str] = []

                for term_id, term in self.linked_knowledge.terms.items():
                    if hasattr(term, "occurrences"):
                        for occ in term.occurrences:
                            if occ.document_id == doc_id and occ.chunk_index == chunk_index:
                                if term.term_type == "acronym":
                                    chunk_acronyms.append(term.term)
                                elif term.term_type == "keyword":
                                    chunk_keywords.append(term.term)
                                break

                html.append(
                    f"""
                    <div class="bg-gray-50 rounded-lg p-4 border border-gray-200 hover:border-gray-300 transition-colors">
                        <div class="flex justify-between items-start mb-3 gap-4">
                            <div class="flex flex-wrap gap-1 items-center" style="max-width: calc(100% - 168px);">
                                <div id="doc-chunk-terms-{
                        chunk_index
                    }" class="flex flex-wrap gap-1 items-center">
                                    {
                        "".join(
                            [
                                f'<span class="inline-block px-2 py-1 text-xs rounded font-medium" style="background-color: #fbbf24; color: #000;">📝 {acronym}</span>'
                                for acronym in chunk_acronyms[:5]
                            ]
                        )
                        if chunk_acronyms
                        else ""
                    }
                                    {
                        "".join(
                            [
                                f'<span class="inline-block px-2 py-1 text-xs rounded font-medium" style="background-color: #1e40af; color: #fff;">🔑 {keyword}</span>'
                                for keyword in chunk_keywords[:5]
                            ]
                        )
                        if chunk_keywords
                        else ""
                    }
                                </div>
                                <div id="doc-chunk-all-terms-{
                        chunk_index
                    }" class="hidden flex-wrap gap-1 items-center">
                                    {
                        "".join(
                            [
                                f'<span class="inline-block px-2 py-1 text-xs rounded font-medium" style="background-color: #fbbf24; color: #000;">📝 {acronym}</span>'
                                for acronym in chunk_acronyms
                            ]
                        )
                        if chunk_acronyms
                        else ""
                    }
                                    {
                        "".join(
                            [
                                f'<span class="inline-block px-2 py-1 text-xs rounded font-medium" style="background-color: #1e40af; color: #fff;">🔑 {keyword}</span>'
                                for keyword in chunk_keywords
                            ]
                        )
                        if chunk_keywords
                        else ""
                    }
                                </div>
                                {
                        f'''<button onclick="toggleDocChunkTerms({chunk_index})"
                                     id="doc-chunk-toggle-{chunk_index}"
                                     class="px-2 py-1 text-xs text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200 rounded cursor-pointer transition-colors">
                                     +{len(chunk_acronyms) - 5 + len(chunk_keywords) - 5 if len(chunk_keywords) > 5 else len(chunk_acronyms) - 5} more
                                </button>'''
                        if len(chunk_acronyms) > 5 or len(chunk_keywords) > 5
                        else ""
                    }
                            </div>
                            <div class="flex gap-2 text-xs flex-shrink-0">
                                {
                        self._render_page_link(
                            doc_id, chunk.page, f"📄 Page {chunk.page}"
                        )
                        if hasattr(chunk, "page")
                        else ""
                    }
                                <span class="px-2 py-1 bg-white rounded text-xs text-gray-500">📦 Chunk {
                        chunk_index + 1
                    }</span>
                                {
                        self._render_pdf_link(doc_id, chunk.page)
                        if hasattr(chunk, "page")
                        else ""
                    }
                            </div>
                        </div>

                        <div class="text-sm text-gray-700 leading-relaxed pt-3">
                            <div id="chunk-text-short-{chunk_index}" class="{
                        "hidden" if False else ""
                    }" data-markdown="{
                        chunk_text_short.replace('"', "&quot;").replace("\n", "\\n")
                    }">
                                {chunk_text_short}
                            </div>
                            <div id="chunk-text-full-{
                        chunk_index
                    }" class="hidden" data-markdown="{
                        chunk_text_full.replace('"', "&quot;").replace("\n", "\\n")
                    }">
                                {chunk_text_full}
                            </div>
                            {
                        f'''<button onclick="toggleChunkText({chunk_index})"
                                     id="chunk-text-toggle-{chunk_index}"
                                     class="mt-2 px-3 py-1 text-sm text-blue-600 hover:text-blue-800 bg-blue-50 hover:bg-blue-100 rounded cursor-pointer transition-colors">
                                     Expand
                                </button>'''
                        if is_truncated
                        else ""
                    }
                        </div>

                        <div class="flex justify-between items-center mt-3 pt-3 border-t border-gray-200">
                            <div class="text-xs text-gray-500">
                                {
                        f"{len(chunk.text)} characters"
                        if hasattr(chunk, "text")
                        else ""
                    }
                            </div>
                            <div class="text-xs text-gray-500">
                                {
                        f"{len(chunk_acronyms)} acronyms, {len(chunk_keywords)} keywords found"
                        if chunk_acronyms or chunk_keywords
                        else "No terms extracted"
                    }
                            </div>
                        </div>
                    </div>
                """
                )

            if not chunk_ids:
                html.append(
                    """
                    <div class="text-gray-500 text-center py-8">
                        No chunks found for this document
                    </div>
                """
                )

            html.append(
                """
                    </div>
                </div>
            </div>
            """
            )

            return "".join(html)

        @router.get("/api/terms", response_class=HTMLResponse)
        async def get_terms(
            search: str = "",
            show_acronyms: bool = True,
            show_keywords: bool = True,
            page: int = Query(1, ge=1),
            per_page: int = Query(20, ge=1, le=100),
        ) -> str:
            """Get filtered and paginated terms list as HTML partial for HTMX."""
            if not self.linked_knowledge:
                return "<div>No data loaded</div>"

            # First filter terms
            filtered_terms = []
            search_lower = search.lower().strip() if search else ""

            for term_id, term in self.linked_knowledge.terms.items():
                is_acronym = term.term_type == "acronym"

                # Apply filters
                if is_acronym and not show_acronyms:
                    continue
                if not is_acronym and not show_keywords:
                    continue

                # Apply search
                if search_lower:
                    term = cast(Term, term)
                    term_text = term.term.lower()
                    full_form = (term.full_form or "").lower()
                    meaning = (term.meaning or "").lower()

                    if search_lower not in term_text and search_lower not in meaning and search_lower not in full_form:
                        continue

                # Add to filtered list
                filtered_terms.append((term_id, term, is_acronym))

            # Paginate filtered results
            total = len(filtered_terms)
            start = (page - 1) * per_page
            end = start + per_page
            terms_page = filtered_terms[start:end]
            total_pages = (total + per_page - 1) // per_page

            # Start wrapper for terms-container
            html = ['<div id="terms-container">']

            # Start terms-list div
            html.append('<div id="terms-list" class="max-h-[720px] overflow-y-auto">')

            # Add count info
            if total > 0:
                html.append(
                    f'<div class="text-sm text-gray-600 mb-3">Showing {start + 1}-{min(end, total)} of {total} terms</div>'
                )

            # Render terms
            for term_id, term, is_acronym in terms_page:
                # Use consistent styling
                badge_style = (
                    "background-color: #fbbf24; color: #000;"
                    if is_acronym
                    else "background-color: #1e40af; color: #fff;"
                )
                icon = "📝" if is_acronym else "🔑"
                term_text = term.term

                # Escape term_id for JavaScript
                escaped_term_id = term_id.replace("'", "\\'").replace('"', '\\"')

                html.append(
                    f"""
                <div class="bg-white rounded-lg shadow p-6 mb-3 hover:shadow-xl transition-all cursor-pointer transform hover:-translate-y-1"
                     hx-get="api/term/{escaped_term_id}"
                     hx-target="#modalContent"
                     hx-trigger="click"
                     hx-on::after-request="document.getElementById('termModal').classList.remove('hidden')">
                    <div class="flex justify-between items-start">
                        <div class="flex-1">
                            <div class="flex items-center gap-3 mb-3">
                                <span class="px-3 py-1.5 text-sm font-semibold rounded-full flex items-center gap-1.5 shadow-sm" style="{
                        badge_style
                    }">
                                    {icon} {"Acronym" if is_acronym else "Keyword"}
                                </span>
                                <h3 class="text-xl font-bold text-gray-900">{
                        term_text
                    }</h3>
                            </div>
                            {
                        f'<p class="text-sm font-semibold text-gray-700 mb-2">Full form: <span class="font-normal">{term.full_form}</span></p>'
                        if is_acronym and hasattr(term, "full_form")
                        else ""
                    }
                            <p class="text-sm text-gray-600 leading-relaxed mb-3">{
                        term.meaning or "No description available"
                    }</p>
                            <div class="flex gap-6 text-sm text-gray-500 border-t pt-3">
                                <span class="flex items-center gap-1">
                                    <span class="font-semibold text-gray-700">{
                        term.total_count if hasattr(term, "total_count") else 0
                    }</span>
                                    <span>occurrences</span>
                                </span>
                                {
                        f'''<span class="flex items-center gap-1">
                                    <span class="font-semibold text-gray-700">{term.mean_score:.3f}</span>
                                    <span>avg score</span>
                                </span>'''
                        if not is_acronym and hasattr(term, "mean_score")
                        else ""
                    }
                            </div>
                        </div>
                        <div class="text-gray-400">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                            </svg>
                        </div>
                    </div>
                </div>
                """
                )

            # Close terms-list div
            if not terms_page:
                html.append("<div class='text-gray-500 text-center py-8'>No terms found matching your criteria</div>")

            html.append("</div>")  # Close terms-list

            # Add pagination controls in separate div
            html.append('<div id="terms-pagination" class="mt-4">')

            if total_pages > 1:
                html.append('<div class="flex justify-center items-center gap-2">')

                # Previous button
                if page > 1:
                    html.append(
                        f"""
                        <button class="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300"
                                onclick="updateTermsFilter({page - 1})">
                            ← Previous
                        </button>
                    """
                    )

                # Page numbers
                for p in range(max(1, page - 2), min(total_pages + 1, page + 3)):
                    if p == page:
                        html.append(f'<span class="px-3 py-1 text-sm bg-yellow-400 rounded font-semibold">{p}</span>')
                    else:
                        html.append(
                            f"""
                            <button class="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300"
                                    onclick="updateTermsFilter({p})">
                                {p}
                            </button>
                        """
                        )

                # Next button
                if page < total_pages:
                    html.append(
                        f"""
                        <button class="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300"
                                onclick="updateTermsFilter({page + 1})">
                            Next →
                        </button>
                    """
                    )

                html.append("</div>")

            html.append("</div>")  # Close terms-pagination
            html.append("</div>")  # Close terms-container

            return "".join(html)

        @router.get("/api/term/{term_id}", response_class=HTMLResponse)
        async def get_term_details(term_id: str) -> str:
            """Get detailed information about a term or acronym."""
            if not self.linked_knowledge or term_id not in self.linked_knowledge.terms:
                return "<div class='text-red-500'>Term not found</div>"

            term = self.linked_knowledge.terms[term_id]
            is_acronym = term.term_type == "acronym"

            # Build markdown content
            markdown_parts = []

            # Header
            if is_acronym:
                markdown_parts.append(f"## Acronym: {term.term}")
                if hasattr(term, "full_form"):
                    markdown_parts.append(f"**Full Form:** {term.full_form}")
            else:
                markdown_parts.append(f"## Keyword: {term.term}")

            markdown_parts.append(f"\n**Meaning:** {term.meaning or 'No description available'}")

            # Statistics
            markdown_parts.append("\n### Statistics")
            markdown_parts.append(f"- **Total Occurrences:** {term.total_count if hasattr(term, 'total_count') else 0}")
            if not is_acronym and hasattr(term, "mean_score"):
                markdown_parts.append(f"- **Mean Score:** {term.mean_score:.3f}")

            # Occurrences
            if term.occurrences:
                markdown_parts.append(f"\n### Occurrences ({len(term.occurrences)} total)")
                for i, occ in enumerate(term.occurrences, 1):
                    doc_name = (
                        self.linked_knowledge.documents[occ.document_id].filename
                        if occ.document_id in self.linked_knowledge.documents
                        else occ.document_id
                    )
                    markdown_parts.append(f"{i}. **{doc_name}** - Page {occ.page}, Chunk {occ.chunk_index}")

                    # Get chunk text if available
                    if occ.document_id in self.linked_knowledge.document_to_chunk_ids_index:
                        chunk_ids = self.linked_knowledge.document_to_chunk_ids_index[occ.document_id]
                        for chunk_id in chunk_ids:
                            chunk = self.linked_knowledge.chunks[chunk_id]
                            if chunk.index == occ.chunk_index:
                                # Show excerpt
                                excerpt = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                                markdown_parts.append(f"   > {excerpt}")
                                break

            # Cooccurrences
            if hasattr(term, "cooccurrences") and term.cooccurrences:
                markdown_parts.append(f"\n### Co-occurring Terms ({len(term.cooccurrences)} total)")
                for cooc in sorted(term.cooccurrences, key=lambda x: x.frequency, reverse=True):
                    markdown_parts.append(
                        f"- **{cooc.term}** (appears together {cooc.frequency} times, confidence: {cooc.confidence:.2f})"
                    )

            # Prepare markdown content for client-side rendering
            markdown_text = "\n".join(markdown_parts)

            # Escape for HTML attribute
            escaped_markdown = markdown_text.replace('"', "&quot;").replace("\n", "\\n").replace("\r", "")

            # Return container with markdown data attribute for client-side rendering
            html_content = f'<div data-markdown="{escaped_markdown}">{markdown_text}</div>'

            import re

            # Note: Table conversion will be handled by marked.js on client-side
            def convert_markdown_table(match: re.Match[str]) -> str:
                table_text = match.group(1)
                lines = table_text.strip().split("\n")
                if len(lines) < 2:
                    return table_text

                html_table = [
                    '<div class="overflow-x-auto my-4"><table class="min-w-full divide-y divide-gray-200 border border-gray-300">'
                ]

                # Header
                header_cells = [cell.strip() for cell in lines[0].split("|") if cell.strip()]
                html_table.append('<thead class="bg-gray-50">')
                html_table.append("<tr>")
                for cell in header_cells:
                    html_table.append(
                        f'<th class="px-3 py-2 text-left text-xs font-medium text-gray-700 uppercase tracking-wider border-r border-gray-200">{cell}</th>'
                    )
                html_table.append("</tr>")
                html_table.append("</thead>")

                # Body (skip separator line)
                if len(lines) > 2:
                    html_table.append('<tbody class="bg-white divide-y divide-gray-200">')
                    for line in lines[2:]:
                        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
                        html_table.append('<tr class="hover:bg-gray-50">')
                        for cell in cells:
                            html_table.append(
                                f'<td class="px-3 py-2 text-sm text-gray-600 border-r border-gray-200">{cell}</td>'
                            )
                        html_table.append("</tr>")
                    html_table.append("</tbody>")

                html_table.append("</table></div>")
                return "".join(html_table)

            # Convert tables
            html_content = re.sub(r"(\|[^\n]+\|(?:\n\|[^\n]+\|)+)", convert_markdown_table, html_content)

            # Headers with better Tailwind classes
            html_content = html_content.replace(
                "### ",
                "<h3 class='text-lg font-bold text-gray-900 mt-6 mb-3 border-b pb-2'>",
            )
            html_content = html_content.replace(
                "## ",
                "<h2 class='text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2'>",
            )
            html_content = html_content.replace("\n<h", "</p>\n<h")
            html_content = html_content.replace("</h3>\n", "</h3>\n<div class='text-sm text-gray-700 space-y-2'>")
            html_content = html_content.replace("</h2>\n", "</h2>\n<div class='text-sm text-gray-700 space-y-2'>")

            # Bold text with Tailwind
            html_content = re.sub(
                r"\*\*([^*]+)\*\*",
                r'<span class="font-semibold text-gray-900">\1</span>',
                html_content,
            )

            # Lists with better styling
            html_content = re.sub(
                r"^- (.+)$",
                r'<li class="flex items-start ml-4"><span class="text-yellow-500 mr-2">•</span><span>\1</span></li>',
                html_content,
                flags=re.MULTILINE,
            )
            html_content = re.sub(
                r"^(\d+)\. (.+)$",
                r'<li class="flex items-start ml-4"><span class="font-semibold text-gray-900 mr-2">\1.</span><span>\2</span></li>',
                html_content,
                flags=re.MULTILINE,
            )

            # Wrap consecutive list items in ul/ol
            html_content = re.sub(
                r'(<li class="flex items-start[^>]*>.*?</li>\n?)+',
                r'<ul class="space-y-1 my-3">\g<0></ul>',
                html_content,
                flags=re.DOTALL,
            )

            # Blockquotes with better styling
            html_content = re.sub(
                r"^   > (.+)$",
                r'<blockquote class="ml-8 pl-4 py-2 border-l-4 border-yellow-400 bg-gray-50 italic text-gray-700 text-sm my-3 rounded-r">\1</blockquote>',
                html_content,
                flags=re.MULTILINE,
            )

            # Statistics section special styling
            html_content = re.sub(
                r"<h3([^>]*)>Statistics</h3>",
                r'<h3\1><span class="inline-block px-2 py-1 bg-yellow-400 text-black rounded text-sm mr-2">📊</span>Statistics</h3>',
                html_content,
            )
            html_content = re.sub(
                r"<h3([^>]*)>Occurrences",
                r'<h3\1><span class="inline-block px-2 py-1 bg-green-100 text-green-800 rounded text-sm mr-2">📍</span>Occurrences',
                html_content,
            )
            html_content = re.sub(
                r"<h3([^>]*)>Co-occurring Terms",
                r'<h3\1><span class="inline-block px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm mr-2">🔗</span>Co-occurring Terms',
                html_content,
            )

            # Wrap in proper containers
            html_content = "<div class='prose prose-sm max-w-none'>" + html_content + "</div>"
            html_content = html_content.replace("\n\n", "</div>\n<div class='text-sm text-gray-700 space-y-2 mt-4'>")
            html_content = html_content.replace("<div class='text-sm text-gray-700 space-y-2'></div>", "")

            return f"""<div class="max-h-[500px] overflow-y-scroll px-2">{html_content}</div>"""

        @router.post("/api/search", response_class=HTMLResponse)
        async def perform_search(
            query: str = Form(...),
            use_optimized: str = Form("true"),
            show_comparison: str = Form("false"),
        ) -> str:
            """Perform search with optional comparison."""
            if not self.search_core:
                return "<div class='text-red-500'>Search not available - no data loaded</div>"

            if not query.strip():
                return "<div class='text-gray-500'>Please enter a search query</div>"

            html = []
            show_comparison_bool: bool = show_comparison == "true"

            if show_comparison_bool:
                # Show side-by-side comparison - enhanced search vs pure semantic
                enhanced_results: List[KnowledgeSearchResult] = self.search_core.search_enhanced(query, k=5)
                # For comparison, similarity search with higher threshold
                semantic_results: List[SimilaritySearchResult] = self.search_core.search_similarity(
                    query, k=5, threshold=0.3
                )

                html.append(
                    """
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- Enhanced Search Results -->
                    <div>
                        <h3 class="text-lg font-semibold mb-3 text-green-700">🚀 Enhanced Search (Terms + Context)</h3>
                        <div class="space-y-3">
                """
                )

                if enhanced_results:
                    for i, result in enumerate(enhanced_results, 1):
                        match_type = result.metadata.get("match_type", "unknown")
                        badge_color = "bg-green-500" if match_type == "direct_keyword" else "bg-blue-500"
                        badge_text = "Keyword Match" if match_type == "direct_keyword" else "Semantic Match"

                        html.append(
                            f"""
                            <div class="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                                <div class="flex justify-between items-start mb-2">
                                    <span class="font-semibold text-sm">Result #{
                                i
                            }</span>
                                    <span class="px-2 py-1 text-xs text-white rounded {
                                badge_color
                            }">{badge_text}</span>
                                </div>
                                <div class="text-xs text-gray-600 mb-2">
                                    📄 {result.document_name} | {
                                self._render_page_link(
                                    result.document_id,
                                    result.page,
                                    f"Page {result.page}",
                                )
                                if result.page and hasattr(result, "document_id")
                                else f"Page {result.page or 'N/A'}"
                            } | Chunk {result.chunk_index or "N/A"}
                                </div>
                                <div class="text-sm text-gray-700">
                                    <div id="search-text-short-k{
                                i
                            }" class="line-clamp-3">
                                        {result.text[:200]}...
                                    </div>
                                    <div id="search-text-full-k{i}" class="hidden">
                                        {result.text}
                                    </div>
                                    {
                                f'''<button onclick="toggleSearchText('k{i}')"
                                             id="search-text-toggle-k{i}"
                                             class="mt-1 px-2 py-0.5 text-xs text-blue-600 hover:text-blue-800 bg-blue-50 hover:bg-blue-100 rounded cursor-pointer transition-colors">
                                             Show more
                                        </button>'''
                                if len(result.text) > 200
                                else ""
                            }
                                </div>
                                <div class="mt-2 text-xs text-gray-500">
                                    Score: {result.score:.3f}
                                </div>
                                {self._render_terms_in_result(result)}
                            </div>
                        """
                        )
                else:
                    html.append('<div class="text-gray-500 text-center py-4">No results found</div>')

                html.append(
                    """
                        </div>
                    </div>

                    <!-- Semantic Results -->
                    <div>
                        <h3 class="text-lg font-semibold mb-3 text-blue-700">🔍 Semantic Search</h3>
                        <div class="space-y-3">
                """
                )

                if semantic_results:
                    for i, sem_result in enumerate(semantic_results, 1):
                        html.append(
                            f"""
                            <div class="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                                <div class="flex justify-between items-start mb-2">
                                    <span class="font-semibold text-sm">Result #{
                                i
                            }</span>
                                    <span class="px-2 py-1 text-xs text-white rounded bg-purple-500">Embeddings</span>
                                </div>
                                <div class="text-xs text-gray-600 mb-2">
                                    📄 {sem_result.document_name} | {
                                self._render_page_link(
                                    sem_result.document_id,
                                    sem_result.page,
                                    f"Page {sem_result.page}",
                                )
                                if sem_result.page
                                and hasattr(sem_result, "document_id")
                                else f"Page {sem_result.page or 'N/A'}"
                            } | Chunk {sem_result.chunk_index or "N/A"}
                                </div>
                                <div class="text-sm text-gray-700">
                                    <div id="search-text-short-k{
                                i
                            }" class="line-clamp-3">
                                        {sem_result.text[:200]}...
                                    </div>
                                    <div id="search-text-full-k{i}" class="hidden">
                                        {sem_result.text}
                                    </div>
                                    {
                                f'''<button onclick="toggleSearchText('k{i}')"
                                             id="search-text-toggle-k{i}"
                                             class="mt-1 px-2 py-0.5 text-xs text-blue-600 hover:text-blue-800 bg-blue-50 hover:bg-blue-100 rounded cursor-pointer transition-colors">
                                             Show more
                                        </button>'''
                                if len(sem_result.text) > 200
                                else ""
                            }
                                </div>
                                <div class="mt-2 text-xs text-gray-500">
                                    Score: {sem_result.score:.3f}
                                </div>
                                {self._render_terms_in_result(sem_result)}
                            </div>
                        """
                        )
                else:
                    html.append('<div class="text-gray-500 text-center py-4">No results found</div>')

                html.append(
                    """
                        </div>
                    </div>
                </div>
                """
                )

            else:
                # Single search result
                results: Union[List[KnowledgeSearchResult], List[SimilaritySearchResult]]
                if use_optimized:
                    results = self.search_core.search_enhanced(query, k=10)
                    title = "🚀 Enhanced Search Results"
                else:
                    results = self.search_core.search_similarity(query, k=10, threshold=0.3)
                    title = "🔍 Similarity Search Results"

                html.append(f'<h3 class="text-lg font-semibold mb-3">{title}</h3>')
                html.append('<div class="space-y-3">')

                if results:
                    for i, result_item in enumerate(results, 1):
                        search_result = cast(
                            Union[KnowledgeSearchResult, SimilaritySearchResult],
                            result_item,
                        )
                        # Both result types have metadata attribute
                        match_type = getattr(search_result, "metadata", {}).get("match_type", "unknown")
                        if match_type == "direct_keyword":
                            badge_color = "bg-green-500"
                            badge_text = "Keyword Match"
                        elif match_type == "semantic":
                            badge_color = "bg-blue-500"
                            badge_text = "Semantic Match"
                        else:
                            badge_color = "bg-purple-500"
                            badge_text = "Embeddings"

                        html.append(
                            f"""
                            <div class="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                                <div class="flex justify-between items-start mb-2">
                                    <span class="font-semibold text-sm">Result #{
                                i
                            }</span>
                                    <span class="px-2 py-1 text-xs text-white rounded {
                                badge_color
                            }">{badge_text}</span>
                                </div>
                                <div class="text-xs text-gray-600 mb-2">
                                    📄 {search_result.document_name} | {
                                self._render_page_link(
                                    search_result.document_id,
                                    search_result.page,
                                    f"Page {search_result.page}",
                                )
                                if search_result.page
                                and hasattr(search_result, "document_id")
                                else f"Page {search_result.page or 'N/A'}"
                            } | Chunk {search_result.chunk_index or "N/A"}
                                </div>
                                <div class="text-sm text-gray-700">
                                    <div id="search-text-short-c{i}" class="">
                                        {search_result.text[:300]}...
                                    </div>
                                    <div id="search-text-full-c{i}" class="hidden">
                                        {search_result.text}
                                    </div>
                                    {
                                f'''<button onclick="toggleSearchText('c{i}')"
                                             id="search-text-toggle-c{i}"
                                             class="mt-1 px-2 py-0.5 text-xs text-blue-600 hover:text-blue-800 bg-blue-50 hover:bg-blue-100 rounded cursor-pointer transition-colors">
                                             Show more
                                        </button>'''
                                if len(search_result.text) > 300
                                else ""
                            }
                                </div>
                                <div class="mt-2 text-xs text-gray-500">
                                    Score: {search_result.score:.3f}
                                </div>
                                {self._render_terms_in_result(search_result)}
                            </div>
                        """
                        )
                else:
                    html.append('<div class="text-gray-500 text-center py-8">No results found</div>')

                html.append("</div>")

            return "".join(html)

        @router.get("/api/links", response_class=HTMLResponse)
        async def get_links(page: int = Query(1, ge=1), per_page: int = Query(15, ge=1, le=50)) -> str:
            """Get paginated links list as HTML partial for HTMX."""
            if not self.linked_knowledge:
                return "<div>No data loaded</div>"

            # Paginate links
            all_links = self.linked_knowledge.links
            total = len(all_links)
            start = (page - 1) * per_page
            end = start + per_page
            links_page = all_links[start:end]
            total_pages = (total + per_page - 1) // per_page

            html = []

            # Add count info
            if total > 0:
                html.append(
                    f'<div class="text-sm text-gray-600 mb-3">Showing {start + 1}-{min(end, total)} of {total} relationships</div>'
                )

            for link in links_page:
                score_percent = link.match_score * 100

                html.append(
                    f"""
                <div class="bg-white rounded-lg shadow p-4 mb-3 hover:shadow-lg transition-shadow">
                    <div class="flex items-center gap-3">
                        <span class="px-2 py-1 rounded text-sm font-semibold flex items-center gap-1" style="background-color: #fbbf24; color: #000;">
                            📝 {link.acronym}
                        </span>
                        <span class="px-2 py-1 rounded text-xs font-semibold" style="background-color: #10b981; color: #fff;">
                            🔗 linked to
                        </span>
                        <span class="px-2 py-1 rounded text-sm font-semibold flex items-center gap-1" style="background-color: #1e40af; color: #fff;">
                            🔑 {link.keyword}
                        </span>
                        <div class="ml-auto">
                            <div class="text-sm text-gray-600">Match Score</div>
                            <div class="w-32 bg-gray-200 rounded-full h-2 mt-1">
                                <div class="h-2 rounded-full" style="width: {score_percent}%; background-color: #fc0;"></div>
                            </div>
                            <div class="text-xs text-gray-500 mt-1">{score_percent:.1f}%</div>
                        </div>
                    </div>
                </div>
                """
                )

            # Add pagination controls
            if total_pages > 1:
                html.append('<div class="flex justify-center items-center gap-2 mt-4">')

                # Previous button
                if page > 1:
                    html.append(
                        f"""
                        <button class="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300"
                                hx-get="api/links?page={page - 1}&per_page={per_page}"
                                hx-target="#links-list"
                                hx-swap="innerHTML">
                            ← Previous
                        </button>
                    """
                    )

                # Page numbers
                for p in range(max(1, page - 2), min(total_pages + 1, page + 3)):
                    if p == page:
                        html.append(f'<span class="px-3 py-1 text-sm bg-yellow-400 rounded font-semibold">{p}</span>')
                    else:
                        html.append(
                            f"""
                            <button class="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300"
                                    hx-get="api/links?page={p}&per_page={per_page}"
                                    hx-target="#links-list"
                                    hx-swap="innerHTML">
                                {p}
                            </button>
                        """
                        )

                # Next button
                if page < total_pages:
                    html.append(
                        f"""
                        <button class="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300"
                                hx-get="api/links?page={page + 1}&per_page={per_page}"
                                hx-target="#links-list"
                                hx-swap="innerHTML">
                            Next →
                        </button>
                    """
                    )

                html.append("</div>")

            return "".join(html) if html else "<div class='text-gray-500 text-center py-8'>No links found</div>"

        @router.get("/api/links/graph-data")
        async def get_links_graph_data() -> dict[str, Any]:
            """Get links data formatted for D3.js graph visualization."""
            if not self.linked_knowledge:
                return {"nodes": [], "links": []}

            # Build nodes and links for D3
            nodes: List[Dict[str, Any]] = []
            edges = []
            node_map = {}

            # Process all links to build unique nodes
            for link in self.linked_knowledge.links:
                # Add acronym node if not exists
                if link.acronym not in node_map:
                    node_map[link.acronym] = len(nodes)
                    nodes.append({"id": link.acronym, "type": "acronym", "label": link.acronym})

                # Add keyword node if not exists
                if link.keyword not in node_map:
                    node_map[link.keyword] = len(nodes)
                    nodes.append({"id": link.keyword, "type": "keyword", "label": link.keyword})

                # Add edge
                edges.append(
                    {
                        "source": link.acronym,
                        "target": link.keyword,
                        "score": link.match_score,
                    }
                )

            return {"nodes": nodes, "links": edges}

    def _render_pdf_link(self, doc_id: str, page: int) -> str:
        """Render a link to open PDF at specific page.

        Args:
            doc_id: Document ID
            page: Page number

        Returns:
            HTML for PDF link button
        """
        doc_filename = self._get_document_filename(doc_id)
        if doc_filename and doc_filename.lower().endswith(".pdf"):
            # Create link to PDF with page anchor
            pdf_url = f"/documents/{doc_filename}#page={page}"
            return f"""
                <a href="{pdf_url}" target="_blank"
                   class="px-2 py-1 bg-yellow-400 text-black rounded text-xs hover:bg-yellow-500 transition-colors">
                    📖 View PDF
                </a>
            """
        return ""

    def _render_page_link(self, doc_id: str, page: int, text: str) -> str:
        """Render a clickable page number that links to the specific page in the document.

        Args:
            doc_id: Document ID
            page: Page number
            text: Text to display in the link

        Returns:
            HTML for clickable page link
        """
        doc_filename = self._get_document_filename(doc_id)
        if doc_filename and doc_filename.lower().endswith(".pdf"):
            # Create link to PDF with page anchor
            pdf_url = f"/documents/{doc_filename}#page={page}"
            return f"""
                <a href="{pdf_url}" target="_blank"
                   class="px-2 py-1 bg-white rounded text-xs hover:bg-gray-200 transition-colors"
                   title="Click to view page {page} in PDF">
                    {text}
                </a>
            """
        else:
            # For non-PDF documents or when filename is not available, return non-clickable span
            return f"<span class='px-2 py-1 bg-white rounded text-xs'>{text}</span>"

    def _render_full_document_link(self, doc_id: str) -> str:
        """Render a link to open the full document.

        Args:
            doc_id: Document ID

        Returns:
            HTML for document link button
        """
        doc_filename = self._get_document_filename(doc_id)
        if doc_filename:
            if doc_filename.lower().endswith(".pdf"):
                doc_url = f"/documents/{doc_filename}"
                return f"""
                    <a href="{doc_url}" target="_blank"
                       class="px-3 py-1 bg-yellow-400 text-black rounded hover:bg-yellow-500 transition-colors">
                        📄 View Full PDF
                    </a>
                """
            else:
                # For text files, serve them directly
                doc_url = f"/documents/{doc_filename}"
                return f"""
                    <a href="{doc_url}" target="_blank"
                       class="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors">
                        📄 View Document
                    </a>
                """
        return ""

    def _render_terms_in_result(self, result: Union[KnowledgeSearchResult, SimilaritySearchResult]) -> str:
        """Render terms found in a search result."""
        html = []

        if result.primary_terms:
            html.append('<div class="mt-3 pt-3 border-t border-gray-200">')
            html.append('<div class="text-xs font-semibold text-gray-600 mb-1">Terms Found:</div>')
            html.append('<div class="flex flex-wrap gap-1">')

            for term in result.primary_terms[:5]:
                if hasattr(term, "acronym"):
                    html.append(
                        f'<span class="px-2 py-1 text-xs rounded" style="background-color: #fbbf24; color: #000;">📝 {getattr(term, "acronym")}</span>'
                    )
                elif hasattr(term, "term"):
                    html.append(
                        f'<span class="px-2 py-1 text-xs rounded" style="background-color: #1e40af; color: #fff;">🔑 {getattr(term, "term")}</span>'
                    )

            if len(result.primary_terms) > 5:
                html.append(f'<span class="text-xs text-gray-500">+{len(result.primary_terms) - 5} more</span>')

            html.append("</div>")
            html.append("</div>")

        return "".join(html)

    def _render_main_page(self) -> str:
        """Render the main HTML page with HTMX and Tailwind."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Visualization</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        [x-cloak] { display: none !important; }
        .node { cursor: pointer; }
        .link { fill: none; stroke-opacity: 0.6; }
        .node-label { pointer-events: none; font-size: 10px; }
    </style>
</head>
<body class="bg-gray-50">
    <div class="min-h-screen">
        <!-- Stats Dashboard -->
        <div class="container mx-auto px-4 py-4">
            <div id="stats" hx-get="api/stats" hx-trigger="load" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                <div class="bg-white rounded-lg shadow p-2 text-center">
                    <div class="text-lg font-bold text-gray-800">-</div>
                    <div class="text-xs text-gray-600">Documents</div>
                </div>
                <div class="bg-white rounded-lg shadow p-2 text-center">
                    <div class="text-lg font-bold text-gray-800">-</div>
                    <div class="text-xs text-gray-600">Terms</div>
                </div>
                <div class="bg-white rounded-lg shadow p-2 text-center">
                    <div class="text-lg font-bold text-gray-800">-</div>
                    <div class="text-xs text-gray-600">Keywords</div>
                </div>
                <div class="bg-white rounded-lg shadow p-2 text-center">
                    <div class="text-lg font-bold text-gray-800">-</div>
                    <div class="text-xs text-gray-600">Acronyms</div>
                </div>
                <div class="bg-white rounded-lg shadow p-2 text-center">
                    <div class="text-lg font-bold text-gray-800">-</div>
                    <div class="text-xs text-gray-600">Links</div>
                </div>
                <div class="bg-white rounded-lg shadow p-2 text-center">
                    <div class="text-lg font-bold text-gray-800">-</div>
                    <div class="text-xs text-gray-600">Chunks</div>
                </div>
            </div>
        </div>

        <!-- Tabs -->
        <div class="container mx-auto px-4">
            <div class="border-b border-gray-200">
                <nav class="-mb-px flex space-x-8">
                    <button onclick="showTab('documents')" id="tab-documents" class="tab-button py-2 px-3 border-b-2 font-medium text-sm" style="border-color: #d4a000; color: #d4a000;">
                        📄 Documents
                    </button>
                    <button onclick="showTab('terms')" id="tab-terms" class="tab-button py-2 px-3 border-b-2 font-medium text-sm border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300">
                        📚 Terms & Acronyms
                    </button>
                    <button onclick="showTab('links')" id="tab-links" class="tab-button py-2 px-3 border-b-2 font-medium text-sm border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300">
                        🔗 Links
                    </button>
                    <button onclick="showTab('search')" id="tab-search" class="tab-button py-2 px-3 border-b-2 font-medium text-sm border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300">
                        🔍 Search
                    </button>
                </nav>
            </div>

            <!-- Tab Content -->
            <div class="py-6">
                <!-- Documents Tab -->
                <div id="documents-content" class="tab-content">
                    <div id="document-details">
                        <div id="documents-list" hx-get="api/documents" hx-trigger="load" class="max-h-[720px] overflow-y-hidden">
                            <div class="animate-pulse">Loading documents...</div>
                        </div>
                    </div>
                </div>

                <!-- Terms Tab -->
                <div id="terms-content" class="tab-content hidden">
                    <div class="mb-4">
                        <div class="flex gap-4 items-center">
                            <input type="text"
                                   name="search"
                                   id="search-terms"
                                   placeholder="Search terms..."
                                   class="px-4 py-2 border rounded-lg flex-1 max-w-md"
                                   onkeyup="updateTermsFilter()">
                            <label class="flex items-center">
                                <input type="checkbox" id="show_acronyms" checked class="mr-2"
                                       onchange="updateTermsFilter()">
                                Show Acronyms
                            </label>
                            <label class="flex items-center">
                                <input type="checkbox" id="show_keywords" checked class="mr-2"
                                       onchange="updateTermsFilter()">
                                Show Keywords
                            </label>
                        </div>
                    </div>
                    <div id="terms-container">
                        <div id="terms-list" hx-get="api/terms" hx-trigger="load" class="max-h-[720px] overflow-y-hidden">
                            <div class="animate-pulse">Loading terms...</div>
                        </div>
                        <div id="terms-pagination" class="mt-4">
                            <!-- Pagination will be loaded here -->
                        </div>
                    </div>
                </div>

                <!-- Links Tab -->
                <div id="links-content" class="tab-content hidden">
                    <!-- View toggle buttons -->
                    <div class="mb-4 flex gap-2">
                        <button onclick="showLinksView('graph')" id="links-view-graph"
                                class="px-4 py-2 rounded font-medium text-sm"
                                style="background-color: #d4a000; color: white;">
                            📊 Graph View
                        </button>
                        <button onclick="showLinksView('list')" id="links-view-list"
                                class="px-4 py-2 bg-gray-200 rounded font-medium text-sm text-gray-700 hover:bg-gray-300">
                            📋 List View
                        </button>
                    </div>

                    <!-- Graph View -->
                    <div id="links-graph-view">
                        <!-- Controls -->
                        <div class="bg-white rounded-lg shadow p-4 mb-4">
                            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <!-- Search -->
                                <div>
                                    <label class="block text-sm font-medium text-gray-700 mb-1">Search & Focus Term:</label>
                                    <input type="text"
                                           id="graph-search"
                                           placeholder="Type to search terms..."
                                           class="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                                           list="term-suggestions">
                                    <datalist id="term-suggestions"></datalist>
                                </div>

                                <!-- Connection Strength Filter -->
                                <div>
                                    <label class="block text-sm font-medium text-gray-700 mb-1">
                                        Min Connection Strength: <span id="strength-value">30%</span>
                                    </label>
                                    <input type="range"
                                           id="strength-filter"
                                           min="0"
                                           max="100"
                                           value="30"
                                           class="w-full">
                                </div>

                                <!-- View Options -->
                                <div>
                                    <label class="block text-sm font-medium text-gray-700 mb-1">Display:</label>
                                    <div class="flex gap-2">
                                        <button onclick="resetGraphView()"
                                                class="px-3 py-1 bg-gray-200 rounded text-sm hover:bg-gray-300">
                                            🔄 Reset View
                                        </button>
                                        <button onclick="showTopConnected()"
                                                class="px-3 py-1 bg-gray-200 rounded text-sm hover:bg-gray-300">
                                            ⭐ Top Connected
                                        </button>
                                    </div>
                                </div>
                            </div>

                            <!-- Legend -->
                            <div class="flex items-center gap-4 text-xs mt-3 pt-3 border-t">
                                <span class="font-semibold">Legend:</span>
                                <div class="flex items-center gap-1">
                                    <div class="w-3 h-3 rounded-full" style="background-color: #fbbf24;"></div>
                                    <span>Acronym</span>
                                </div>
                                <div class="flex items-center gap-1">
                                    <div class="w-3 h-3 rounded-full" style="background-color: #1e40af;"></div>
                                    <span>Keyword</span>
                                </div>
                                <div class="flex items-center gap-1">
                                    <div class="w-3 h-3 rounded-full" style="background-color: #10b981;"></div>
                                    <span>Selected</span>
                                </div>
                                <span class="text-gray-500 ml-auto">Click nodes to explore connections</span>
                            </div>
                        </div>

                        <!-- Graph Container -->
                        <div id="links-graph" class="bg-white rounded-lg shadow" style="height: 600px; position: relative;">
                            <div id="graph-info" class="absolute top-2 left-2 bg-white/90 px-3 py-2 rounded shadow text-sm z-10"></div>
                        </div>
                    </div>

                    <!-- List View (hidden by default) -->
                    <div id="links-list-view" class="hidden">
                        <div id="links-list" hx-get="api/links" hx-trigger="load" class="max-h-[720px] overflow-y-hidden">
                            <div class="animate-pulse">Loading links...</div>
                        </div>
                    </div>
                </div>

                <!-- Search Tab -->
                <div id="search-content" class="tab-content hidden">
                    <div class="bg-white rounded-lg shadow-sm p-6 mb-6">
                        <form hx-post="api/search"
                              hx-target="#search-results"
                              hx-indicator="#search-spinner">

                            <div class="mb-4">
                                <label class="block text-sm font-medium text-gray-700 mb-2">
                                    Enter your search query:
                                </label>
                                <input type="text"
                                       name="query"
                                       class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-yellow-400 focus:border-transparent"
                                       required>
                            </div>

                            <div class="flex flex-wrap gap-4 mb-4">
                                <label class="flex items-center">
                                    <input type="radio" name="use_optimized" value="true" checked class="mr-2">
                                    <span class="text-sm">🚀 Optimized Search (Keywords + Embeddings)</span>
                                </label>
                                <label class="flex items-center">
                                    <input type="radio" name="use_optimized" value="false" class="mr-2">
                                    <span class="text-sm">🔍 Embeddings-Only Search</span>
                                </label>
                            </div>

                            <div class="mb-4">
                                <label class="flex items-center">
                                    <input type="checkbox" name="show_comparison" value="true" class="mr-2">
                                    <span class="text-sm font-medium">Show side-by-side comparison</span>
                                </label>
                            </div>

                            <button type="submit"
                                    class="px-6 py-2 text-black font-semibold rounded-lg hover:opacity-80 transition-opacity"
                                    style="background-color: #fc0;">
                                Search
                            </button>

                            <span id="search-spinner" class="htmx-indicator ml-4">
                                <span class="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-500"></span>
                                Searching...
                            </span>
                        </form>
                    </div>

                    <div class="bg-gray-50 rounded-lg p-4 mb-4">
                        <h3 class="text-sm font-semibold text-gray-700 mb-2">Search Tips:</h3>
                        <ul class="text-xs text-gray-600 space-y-1">
                            <li>• <b>Optimized Search</b>: First looks for exact keyword/acronym matches, then uses semantic search</li>
                            <li>• <b>Embeddings-Only</b>: Uses only semantic similarity to find relevant content</li>
                            <li>• <b>Side-by-side comparison</b>: Shows how both methods perform on the same query</li>
                        </ul>
                    </div>

                    <div id="search-results" class="max-h-[720px] overflow-y-hidden">
                        <!-- Search results will be loaded here -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Enhanced Modal with HTMX and Tailwind -->
    <div id="termModal" class="fixed inset-0 bg-black/60 backdrop-blur-sm hidden z-50 transition-all duration-300"
         onclick="closeModalOnBackdrop(event)">
        <div class="flex items-center justify-center min-h-screen p-4">
            <div class="bg-white rounded-xl shadow-2xl max-w-5xl w-full max-h-[85vh] overflow-hidden transform transition-all duration-300 scale-95 opacity-0"
                 id="modalDialog"
                 onclick="event.stopPropagation()">
                <!-- Modal Header -->
                <div class="flex justify-between items-center px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-gray-50 to-white">
                    <h2 class="text-2xl font-bold text-gray-900 flex items-center gap-2">
                        <span class="text-yellow-500">📋</span>
                        Term Details
                    </h2>
                    <button onclick="closeTermModal()"
                            class="p-2 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>
                <!-- Modal Content -->
                <div id="modalContent" class="p-6 bg-gradient-to-b from-white to-gray-50">
                    <div class="flex justify-center items-center py-12">
                        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-yellow-500"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Toggle chunk text visibility (expand/collapse)
        function toggleChunkText(chunkIndex) {
            const shortText = document.getElementById('chunk-text-short-' + chunkIndex);
            const fullText = document.getElementById('chunk-text-full-' + chunkIndex);
            const toggleBtn = document.getElementById('chunk-text-toggle-' + chunkIndex);

            if (shortText.classList.contains('hidden')) {
                // Show short version
                shortText.classList.remove('hidden');
                fullText.classList.add('hidden');
                toggleBtn.textContent = 'Expand';
            } else {
                // Show full text
                shortText.classList.add('hidden');
                fullText.classList.remove('hidden');
                toggleBtn.textContent = 'Collapse';
            }
        }

        // Toggle search result text visibility
        function toggleSearchText(resultId) {
            const shortText = document.getElementById('search-text-short-' + resultId);
            const fullText = document.getElementById('search-text-full-' + resultId);
            const toggleBtn = document.getElementById('search-text-toggle-' + resultId);

            if (shortText.classList.contains('hidden')) {
                // Show short version
                shortText.classList.remove('hidden');
                fullText.classList.add('hidden');
                toggleBtn.textContent = 'Show more';
            } else {
                // Show full text
                shortText.classList.add('hidden');
                fullText.classList.remove('hidden');
                toggleBtn.textContent = 'Show less';
            }
        }

        // Toggle chunk terms visibility
        function toggleChunkTerms(chunkIndex) {
            const shortTerms = document.getElementById('chunk-terms-' + chunkIndex);
            const allTerms = document.getElementById('chunk-all-terms-' + chunkIndex);
            const toggleBtn = document.getElementById('chunk-toggle-' + chunkIndex);

            if (shortTerms.classList.contains('hidden')) {
                // Show short version
                shortTerms.classList.remove('hidden');
                shortTerms.classList.add('flex');
                allTerms.classList.add('hidden');
                allTerms.classList.remove('flex');
                toggleBtn.textContent = toggleBtn.textContent.replace('Show less', '+').replace('+ ', '+');
            } else {
                // Show all terms
                shortTerms.classList.add('hidden');
                shortTerms.classList.remove('flex');
                allTerms.classList.remove('hidden');
                allTerms.classList.add('flex');
                toggleBtn.textContent = 'Show less';
            }
        }

        function toggleDocChunkTerms(chunkIndex) {
            const shortTerms = document.getElementById('doc-chunk-terms-' + chunkIndex);
            const allTerms = document.getElementById('doc-chunk-all-terms-' + chunkIndex);
            const toggleBtn = document.getElementById('doc-chunk-toggle-' + chunkIndex);

            if (shortTerms.classList.contains('hidden')) {
                // Show short version
                shortTerms.classList.remove('hidden');
                shortTerms.classList.add('flex');
                allTerms.classList.add('hidden');
                allTerms.classList.remove('flex');
                toggleBtn.textContent = toggleBtn.textContent.replace('Show less', '+').replace('+ ', '+');
            } else {
                // Show all terms
                shortTerms.classList.add('hidden');
                shortTerms.classList.remove('flex');
                allTerms.classList.remove('hidden');
                allTerms.classList.add('flex');
                toggleBtn.textContent = 'Show less';
            }
        }

        // Initialize D3 graph when links tab is shown
        let graphInitialized = false;
        let graphData = null;
        let simulation = null;
        let selectedNode = null;

        function initializeLinksGraph() {
            if (graphInitialized) return;
            graphInitialized = true;

            // Fetch graph data
            fetch('api/links/graph-data')
                .then(response => response.json())
                .then(data => {
                    if (data.nodes.length === 0) {
                        document.getElementById('links-graph').innerHTML = '<div class="flex items-center justify-center h-full text-gray-500">No links to visualize</div>';
                        return;
                    }

                    graphData = data;

                    // Populate search suggestions
                    const datalist = document.getElementById('term-suggestions');
                    datalist.innerHTML = data.nodes.map(n => `<option value="${n.label}">`).join('');

                    // Setup event listeners
                    document.getElementById('graph-search').addEventListener('input', handleSearch);
                    document.getElementById('strength-filter').addEventListener('input', handleStrengthFilter);

                    // Show only top connected nodes initially
                    showTopConnected();
                });
        }

        function renderGraph(nodes, links, focusNodeId = null) {
            const container = document.getElementById('links-graph');
            const width = container.clientWidth;
            const height = 600;

            // Update info
            const info = document.getElementById('graph-info');
            if (info) {
                info.innerHTML = `Showing ${nodes.length} nodes, ${links.length} connections`;
            }

            // Clear any existing SVG
            d3.select('#links-graph').selectAll('svg').remove();

            // Create SVG with zoom
            const svg = d3.select('#links-graph')
                .append('svg')
                .attr('width', width)
                .attr('height', height);

            const g = svg.append('g');

            // Disable zoom and pan - graph stays centered
            svg.on('wheel.zoom', null)
                .on('mousedown.zoom', null)
                .on('dblclick.zoom', null)
                .on('touchstart.zoom', null)
                .on('touchmove.zoom', null)
                .on('touchend.zoom', null);

            // Color function
            function getNodeColor(d) {
                if (d.id === focusNodeId) return '#10b981'; // Green for selected
                if (d.type === 'acronym') return '#fbbf24';  // Yellow for acronyms
                return '#1e40af';  // Blue for keywords
            }

            // Size function based on connections
            function getNodeSize(d) {
                const connections = links.filter(l => l.source === d.id || l.target === d.id).length;
                return Math.min(20, 8 + connections * 2);
            }

            // Create force simulation with better parameters and bounds
            simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links)
                    .id(d => d.id)
                    .distance(d => 80 * (1 - d.score)))  // Reduced distance
                .force('charge', d3.forceManyBody()
                    .strength(-400)  // Reduced repulsion
                    .distanceMax(200))  // Reduced max distance
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide()
                    .radius(d => getNodeSize(d) + 5))  // Reduced collision radius
                .force('x', d3.forceX(width / 2).strength(0.1))  // Keep nodes horizontally centered
                .force('y', d3.forceY(height / 2).strength(0.1));  // Keep nodes vertically centered

            // Create links
            const link = g.append('g')
                .selectAll('line')
                .data(links)
                .enter().append('line')
                .attr('stroke', d => d3.interpolate('#e5e7eb', '#d4a000')(d.score))
                .attr('stroke-width', d => Math.max(1, d.score * 4))
                .attr('stroke-opacity', 0.6);

            // Create node groups
            const nodeGroup = g.append('g')
                .selectAll('g')
                .data(nodes)
                .enter().append('g')
                .attr('cursor', 'pointer')
                .on('click', (event, d) => {
                    event.stopPropagation();
                    selectNode(d.id);
                })
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));

            // Add circles
            nodeGroup.append('circle')
                .attr('r', d => getNodeSize(d))
                .attr('fill', d => getNodeColor(d))
                .attr('stroke', '#fff')
                .attr('stroke-width', 2)
                .transition()
                .duration(300)
                .attr('r', d => getNodeSize(d));

            // Add labels (only for larger nodes or focused node)
            nodeGroup.append('text')
                .attr('dy', -20)
                .attr('text-anchor', 'middle')
                .text(d => {
                    const connections = links.filter(l => l.source === d.id || l.target === d.id).length;
                    return (connections > 2 || d.id === focusNodeId) ? d.label : '';
                })
                .style('font-size', '11px')
                .style('font-weight', d => d.id === focusNodeId ? 'bold' : 'normal')
                .style('fill', '#333')
                .style('pointer-events', 'none');

            // Add tooltips for all nodes
            nodeGroup.append('title')
                .text(d => {
                    const connections = links.filter(l => l.source === d.id || l.target === d.id).length;
                    return `${d.type === 'acronym' ? '📝 Acronym' : '🔑 Keyword'}: ${d.label}
Connections: ${connections}`;
                });

            // Update positions on tick with boundary constraints
            simulation.on('tick', () => {
                // Keep nodes within bounds
                nodes.forEach(d => {
                    const radius = getNodeSize(d);
                    d.x = Math.max(radius + 20, Math.min(width - radius - 20, d.x));
                    d.y = Math.max(radius + 20, Math.min(height - radius - 20, d.y));
                });

                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                nodeGroup.attr('transform', d => `translate(${d.x},${d.y})`);
            });

            // Drag functions
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }

            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }

            // No auto-zoom - keep everything centered
        }

        function showTopConnected() {
            if (!graphData) return;

            // Count connections for each node
            const connectionCount = {};
            graphData.nodes.forEach(n => connectionCount[n.id] = 0);
            graphData.links.forEach(l => {
                connectionCount[l.source]++;
                connectionCount[l.target]++;
            });

            // Get top 30 most connected nodes
            const topNodes = graphData.nodes
                .sort((a, b) => connectionCount[b.id] - connectionCount[a.id])
                .slice(0, 30);

            const topNodeIds = new Set(topNodes.map(n => n.id));

            // Filter links to only include those between top nodes
            const filteredLinks = graphData.links.filter(l =>
                topNodeIds.has(l.source) && topNodeIds.has(l.target)
            );

            renderGraph(topNodes, filteredLinks);
        }

        function selectNode(nodeId) {
            if (!graphData) return;

            selectedNode = nodeId;
            document.getElementById('graph-search').value = nodeId;

            // Get the selected node and its neighbors
            const neighbors = new Set([nodeId]);
            const neighborLinks = graphData.links.filter(l => {
                if (l.source === nodeId || l.target === nodeId) {
                    neighbors.add(l.source);
                    neighbors.add(l.target);
                    return true;
                }
                return false;
            });

            // Get all neighbor nodes
            const neighborNodes = graphData.nodes.filter(n => neighbors.has(n.id));

            renderGraph(neighborNodes, neighborLinks, nodeId);
        }

        function handleSearch(event) {
            const searchTerm = event.target.value.toLowerCase();
            if (!searchTerm) {
                showTopConnected();
                return;
            }

            // Find matching node
            const matchingNode = graphData.nodes.find(n =>
                n.label.toLowerCase() === searchTerm
            );

            if (matchingNode) {
                selectNode(matchingNode.id);
            }
        }

        function handleStrengthFilter(event) {
            const threshold = event.target.value / 100;
            document.getElementById('strength-value').textContent = `${event.target.value}%`;

            if (!graphData) return;

            // Filter links by strength
            const filteredLinks = graphData.links.filter(l => l.score >= threshold);

            // Get nodes that have at least one connection above threshold
            const connectedNodes = new Set();
            filteredLinks.forEach(l => {
                connectedNodes.add(l.source);
                connectedNodes.add(l.target);
            });

            const filteredNodes = graphData.nodes.filter(n => connectedNodes.has(n.id));

            renderGraph(filteredNodes, filteredLinks, selectedNode);
        }

        function resetGraphView() {
            selectedNode = null;
            document.getElementById('graph-search').value = '';
            document.getElementById('strength-filter').value = 30;
            document.getElementById('strength-value').textContent = '30%';
            showTopConnected();
        }

        // View switching for links
        function showLinksView(view) {
            const graphView = document.getElementById('links-graph-view');
            const listView = document.getElementById('links-list-view');
            const graphBtn = document.getElementById('links-view-graph');
            const listBtn = document.getElementById('links-view-list');

            if (view === 'graph') {
                graphView.classList.remove('hidden');
                listView.classList.add('hidden');
                graphBtn.style.backgroundColor = '#d4a000';
                graphBtn.style.color = 'white';
                graphBtn.classList.remove('bg-gray-200', 'text-gray-700');
                listBtn.style.backgroundColor = '';
                listBtn.style.color = '';
                listBtn.classList.add('bg-gray-200', 'text-gray-700');

                // Initialize graph if not done yet
                initializeLinksGraph();
            } else {
                graphView.classList.add('hidden');
                listView.classList.remove('hidden');
                listBtn.style.backgroundColor = '#d4a000';
                listBtn.style.color = 'white';
                listBtn.classList.remove('bg-gray-200', 'text-gray-700');
                graphBtn.style.backgroundColor = '';
                graphBtn.style.color = '';
                graphBtn.classList.add('bg-gray-200', 'text-gray-700');
            }
        }

        // Update terms filter with debouncing
        let termsFilterTimeout;
        let currentTermsPage = 1;

        function updateTermsFilter(page = 1) {
            clearTimeout(termsFilterTimeout);
            currentTermsPage = page;
            termsFilterTimeout = setTimeout(() => {
                const search = document.getElementById('search-terms').value;
                const showAcronyms = document.getElementById('show_acronyms').checked;
                const showKeywords = document.getElementById('show_keywords').checked;

                const params = new URLSearchParams({
                    search: search,
                    show_acronyms: showAcronyms ? 'true' : 'false',
                    show_keywords: showKeywords ? 'true' : 'false',
                    page: currentTermsPage.toString()
                });

                htmx.ajax('GET', `/api/terms?${params.toString()}`, '#terms-container');
            }, page !== currentTermsPage ? 0 : 500); // No debounce for page changes
        }

        // Tab switching
        function showTab(tabName) {
            // Hide all content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.add('hidden');
            });

            // Reset all tab buttons
            document.querySelectorAll('.tab-button').forEach(button => {
                button.style.borderColor = 'transparent';
                button.style.color = '#6b7280';
                button.classList.add('border-transparent', 'text-gray-500');
            });

            // Show selected content
            document.getElementById(tabName + '-content').classList.remove('hidden');

            // Highlight selected tab with #d4a000 (darker yellow)
            const selectedTab = document.getElementById('tab-' + tabName);
            selectedTab.classList.remove('border-transparent', 'text-gray-500');
            selectedTab.style.borderColor = '#d4a000';
            selectedTab.style.color = '#d4a000';

            // Initialize graph if showing links tab
            if (tabName === 'links' && document.getElementById('links-graph-view').classList.contains('hidden') === false) {
                initializeLinksGraph();
            }
        }

        // Enhanced Modal functions with animations
        function openTermModal(termId) {
            const modal = document.getElementById('termModal');
            const dialog = document.getElementById('modalDialog');

            // Show modal with animation
            modal.classList.remove('hidden');
            setTimeout(() => {
                dialog.classList.remove('scale-95', 'opacity-0');
                dialog.classList.add('scale-100', 'opacity-100');
            }, 10);

            // HTMX will handle content loading automatically
        }

        function closeTermModal() {
            const modal = document.getElementById('termModal');
            const dialog = document.getElementById('modalDialog');

            // Hide with animation
            dialog.classList.remove('scale-100', 'opacity-100');
            dialog.classList.add('scale-95', 'opacity-0');

            setTimeout(() => {
                modal.classList.add('hidden');
            }, 300);
        }

        function closeModalOnBackdrop(event) {
            if (event.target === event.currentTarget) {
                closeTermModal();
            }
        }

        // Removed ESC key handler - using click outside to close instead

        // Auto-open modal after HTMX loads content
        htmx.on('htmx:afterRequest', function(evt) {
            if (evt.detail.target.id === 'modalContent' && evt.detail.successful) {
                const dialog = document.getElementById('modalDialog');
                setTimeout(() => {
                    dialog.classList.remove('scale-95', 'opacity-0');
                    dialog.classList.add('scale-100', 'opacity-100');
                }, 10);
            }
        });

        // Update stats when loaded
        htmx.on('#stats', 'htmx:afterSwap', function(evt) {
            const data = JSON.parse(evt.detail.xhr.response);
            if (data && !data.error) {
                evt.detail.target.innerHTML = `
                    <div class="bg-white rounded-lg shadow p-2 text-center">
                        <div class="text-lg font-bold text-gray-800">${data.total_documents}</div>
                        <div class="text-xs text-gray-600">Documents</div>
                    </div>
                    <div class="bg-white rounded-lg shadow p-2 text-center">
                        <div class="text-lg font-bold text-gray-800">${data.total_terms}</div>
                        <div class="text-xs text-gray-600">Terms</div>
                    </div>
                    <div class="bg-white rounded-lg shadow p-2 text-center">
                        <div class="text-lg font-bold text-gray-800">${data.total_keywords}</div>
                        <div class="text-xs text-gray-600">Keywords</div>
                    </div>
                    <div class="bg-white rounded-lg shadow p-2 text-center">
                        <div class="text-lg font-bold text-gray-800">${data.total_acronyms}</div>
                        <div class="text-xs text-gray-600">Acronyms</div>
                    </div>
                    <div class="bg-white rounded-lg shadow p-2 text-center">
                        <div class="text-lg font-bold text-gray-800">${data.total_links}</div>
                        <div class="text-xs text-gray-600">Links</div>
                    </div>
                    <div class="bg-white rounded-lg shadow p-2 text-center">
                        <div class="text-lg font-bold text-gray-800">${data.total_chunks}</div>
                        <div class="text-xs text-gray-600">Chunks</div>
                    </div>
                `;
            }
        });
    </script>
</body>
</html>
"""
