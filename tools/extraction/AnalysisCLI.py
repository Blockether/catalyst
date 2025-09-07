#!/usr/bin/env python3
"""
Table of Contents Generator CLI.

Generates a hierarchical Table of Contents from batch technical analysis JSON files.
"""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.tree import Tree

console = Console()


class TOCSection(BaseModel):
    """A section in the Table of Contents with hierarchical information."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    level: int
    start_page: int
    end_page: int
    summary: str
    type: str
    text: str
    children: List['TOCSection'] = []
    source_file: str
    batch_number: int


class TermInfo(BaseModel):
    """Information about a term/keyword."""
    term: str
    type: str  # 'acronym' or 'keyword'
    full_form: Optional[str]
    definition: Optional[str]
    page_found: int
    source_file: str
    batch_number: int


class TermSectionMapping(BaseModel):
    """Mapping of a term to sections where it appears."""
    term_infos: List[TermInfo]  # All variations of this term found
    section_ids: List[str]  # List of section UUIDs where any variation appears


class TOCDocument(BaseModel):
    """The complete Table of Contents document."""
    generated_at: str
    document_metadata: Optional[Dict[str, Any]] = None  # Document metadata from batch files
    total_batches: int
    total_sections: int
    total_terms: int
    sections: List[TOCSection]
    terms: Dict[str, TermSectionMapping]  # keyword -> section mapping


class TOCGeneratorCLI:
    """CLI for generating Table of Contents from technical analysis batches."""
    
    def __init__(self, output_dir: Path = Path("output/technical_analysis_responses")):
        """
        Initialize the TOC Generator CLI.
        
        Args:
            output_dir: Directory containing technical analysis batch files
        """
        self.output_dir = output_dir
        self.batch_files: List[Path] = []
        self.all_sections: List[Dict[str, Any]] = []
        self.all_terms: List[Dict[str, Any]] = []
        self.terms_mapping: Dict[str, TermSectionMapping] = {}
        self.section_lookup: Dict[str, TOCSection] = {}  # UUID -> TOCSection mapping
        self.document_metadata: Optional[Dict[str, Any]] = None  # Document metadata from batch files
        
    def _find_batch_files(self) -> None:
        """Find all batch JSON files in the output directory."""
        self.batch_files = sorted(
            self.output_dir.glob("batch_*_pages_*.json"),
            key=lambda x: self._extract_batch_number(x)
        )
        
    @staticmethod
    def _extract_batch_number(file_path: Path) -> int:
        """Extract batch number from filename."""
        try:
            # Format: batch_1_pages_1-2_20250907_170640.json
            parts = file_path.stem.split('_')
            for i, part in enumerate(parts):
                if part == "batch" and i + 1 < len(parts):
                    return int(parts[i + 1])
            return 0
        except (ValueError, IndexError):
            return 0
            
    def _load_batch_files(self) -> None:
        """Load all batch files and extract sections and terms."""
        self.all_sections = []
        self.all_terms = []
        
        for batch_file in self.batch_files:
            try:
                with open(batch_file, 'r') as f:
                    data = json.load(f)
                    
                batch_num = self._extract_batch_number(batch_file)
                
                # Extract document metadata from first batch file
                if self.document_metadata is None and 'document_metadata' in data:
                    self.document_metadata = data['document_metadata']
                
                # Extract sections from the response
                if 'response' in data and 'sections' in data['response']:
                    for section in data['response']['sections']:
                        section_copy = section.copy()
                        section_copy['source_file'] = batch_file.name
                        section_copy['batch_number'] = batch_num
                        self.all_sections.append(section_copy)
                        
                # Extract terms from the response
                if 'response' in data and 'terms' in data['response']:
                    for term in data['response']['terms']:
                        term_copy = term.copy()
                        term_copy['source_file'] = batch_file.name
                        term_copy['batch_number'] = batch_num
                        self.all_terms.append(term_copy)
                        
            except Exception as e:
                console.print(f"[red]Error loading {batch_file}: {e}[/red]")
                
    def _build_hierarchical_toc(self) -> List[TOCSection]:
        """Build hierarchical TOC structure from flat sections list."""
        root_sections: List[TOCSection] = []
        section_stack: List[TOCSection] = []
        self.section_lookup = {}  # Reset lookup
        
        for section_data in self.all_sections:
            section = TOCSection(
                title=section_data['title'],
                level=section_data['level'],
                start_page=section_data['start_page'],
                end_page=section_data['end_page'],
                summary=section_data['summary'],
                type=section_data['type'],
                text=section_data['text'],
                source_file=section_data['source_file'],
                batch_number=section_data['batch_number']
            )
            
            # Store in lookup dictionary
            self.section_lookup[section.id] = section
            
            # Find parent section
            while section_stack and section_stack[-1].level >= section.level:
                section_stack.pop()
                
            if section_stack:
                # Add as child to the last section in stack
                section_stack[-1].children.append(section)
            else:
                # Add as root section
                root_sections.append(section)
                
            section_stack.append(section)
            
        return root_sections
        
    def _build_terms_mapping(self) -> None:
        """Build mapping of terms to sections where they appear."""
        self.terms_mapping = {}
        
        for term_data in self.all_terms:
            term_key = term_data['term'].lower()  # Use lowercase for consistent mapping
            
            # Create TermInfo object
            term_info = TermInfo(
                term=term_data['term'],
                type=term_data['type'],
                full_form=term_data.get('full_form'),
                definition=term_data.get('definition'),
                page_found=term_data['page_found'],
                source_file=term_data['source_file'],
                batch_number=term_data['batch_number']
            )
            
            # Find sections where this term appears (by UUID)
            section_ids_with_term = []
            for section in self.section_lookup.values():
                # Check if term appears in section (based on page range)
                if (section.start_page <= term_data['page_found'] <= section.end_page):
                    section_ids_with_term.append(section.id)
            
            # If term already exists, add this variation
            if term_key in self.terms_mapping:
                # Check if this exact variation already exists
                existing_variations = self.terms_mapping[term_key].term_infos
                variation_exists = False
                
                for existing in existing_variations:
                    if (existing.term == term_info.term and
                        existing.type == term_info.type and
                        existing.full_form == term_info.full_form and
                        existing.definition == term_info.definition):
                        variation_exists = True
                        break
                
                # Add new variation if it doesn't exist
                if not variation_exists:
                    self.terms_mapping[term_key].term_infos.append(term_info)
                    
                # Add new section IDs (avoid duplicates)
                existing_ids = set(self.terms_mapping[term_key].section_ids)
                new_ids = [sid for sid in section_ids_with_term if sid not in existing_ids]
                self.terms_mapping[term_key].section_ids.extend(new_ids)
            else:
                # Create new mapping
                self.terms_mapping[term_key] = TermSectionMapping(
                    term_infos=[term_info],
                    section_ids=section_ids_with_term
                )
        
    def _display_toc_tree(self, sections: List[TOCSection]) -> None:
        """Display TOC as a tree structure."""
        tree = Tree("[bold cyan]Table of Contents[/bold cyan]")
        
        def add_section_to_tree(section: TOCSection, parent_node):
            """Recursively add sections to tree."""
            section_text = f"[bold]{section.title}[/bold] "
            section_text += f"[dim](p{section.start_page}"
            if section.end_page != section.start_page:
                section_text += f"-{section.end_page}"
            section_text += f", {section.type}"
            # Show abbreviated UUID for debugging (first 8 chars)
            section_text += f", id: {section.id[:8]}...)[/dim]"
            
            node = parent_node.add(section_text)
            
            # Add summary as a child node
            if section.summary:
                summary_lines = section.summary.split('\n')
                for line in summary_lines[:2]:  # Show first 2 lines
                    if line.strip():
                        node.add(f"[italic dim]{line.strip()[:80]}...[/italic dim]")
                        
            # Recursively add children
            for child in section.children:
                add_section_to_tree(child, node)
                
        for section in sections:
            add_section_to_tree(section, tree)
            
        console.print(tree)
        
    def _display_terms_summary(self) -> None:
        """Display summary of terms and their section mappings."""
        console.print("\n[bold cyan]Terms & Keywords Summary[/bold cyan]")
        
        # Create table for top terms
        table = Table(title="Top Terms by Frequency")
        table.add_column("Term", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Sections", style="green")
        table.add_column("Pages", style="magenta")
        
        # Sort terms by number of sections they appear in
        sorted_terms = sorted(
            self.terms_mapping.items(),
            key=lambda x: len(x[1].section_ids),
            reverse=True
        )[:10]  # Show top 10
        
        for term_key, mapping in sorted_terms:
            section_count = len(mapping.section_ids)
            
            # Get primary term info (first variation)
            primary_info = mapping.term_infos[0]
            
            # Collect all unique definitions/full forms
            all_full_forms = []
            all_pages = set()
            term_types = set()
            
            for info in mapping.term_infos:
                if info.type == "acronym" and info.full_form and info.full_form not in all_full_forms:
                    all_full_forms.append(info.full_form)
                all_pages.add(info.page_found)
                term_types.add(info.type)
            
            # Format display term
            display_term = primary_info.term
            if all_full_forms:
                display_term += f" ({'; '.join(all_full_forms)})"
            
            # Show all page numbers where term appears
            pages = "p" + ", p".join(map(str, sorted(all_pages)))
            
            # Show term type(s)
            term_type_str = "/".join(sorted(term_types))
            
            table.add_row(
                display_term,
                term_type_str,
                str(section_count),
                pages
            )
            
        console.print(table)
        
    def _display_statistics(self) -> None:
        """Display statistics about the TOC."""
        table = Table(title="TOC Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Batch Files", str(len(self.batch_files)))
        table.add_row("Total Sections", str(len(self.all_sections)))
        table.add_row("Total Terms", str(len(self.all_terms)))
        table.add_row("Unique Terms", str(len(self.terms_mapping)))
        
        # Count by level
        level_counts = {}
        for section in self.all_sections:
            level = section['level']
            level_counts[level] = level_counts.get(level, 0) + 1
            
        for level in sorted(level_counts.keys()):
            table.add_row(f"Level {level} Sections", str(level_counts[level]))
            
        # Count by section type
        type_counts = {}
        for section in self.all_sections:
            section_type = section['type']
            type_counts[section_type] = type_counts.get(section_type, 0) + 1
            
        for section_type in sorted(type_counts.keys()):
            table.add_row(f"{section_type.capitalize()} Sections", str(type_counts[section_type]))
            
        # Count by term type
        term_type_counts = {'acronym': 0, 'keyword': 0}
        for term in self.all_terms:
            term_type = term.get('type', 'keyword')
            term_type_counts[term_type] = term_type_counts.get(term_type, 0) + 1
            
        for term_type, count in term_type_counts.items():
            if count > 0:
                table.add_row(f"{term_type.capitalize()}s", str(count))
            
        console.print(table)
        
        # Show terms with multiple variations
        terms_with_variations = [
            (key, mapping) for key, mapping in self.terms_mapping.items() 
            if len(mapping.term_infos) > 1
        ]
        
        if terms_with_variations:
            console.print(f"\n[yellow]Note: {len(terms_with_variations)} terms have multiple variations/definitions[/yellow]")
        
    def _save_toc(self, toc_sections: List[TOCSection]) -> Path:
        """Save TOC to JSON file."""
        toc_doc = TOCDocument(
            generated_at=datetime.now().isoformat(),
            document_metadata=self.document_metadata,
            total_batches=len(self.batch_files),
            total_sections=len(self.all_sections),
            total_terms=len(self.all_terms),
            sections=toc_sections,
            terms=self.terms_mapping
        )
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"table_of_contents_{timestamp}.json"
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(toc_doc.model_dump(), f, indent=2)
            
        return output_file
        
    def _generate_toc(self) -> None:
        """Generate the Table of Contents."""
        # Find and load batch files
        self._find_batch_files()
        
        if not self.batch_files:
            console.print("[red]No batch files found in output directory[/red]")
            return
            
        console.print(f"[green]Found {len(self.batch_files)} batch files[/green]")
        
        # Load all sections
        with console.status("Loading batch files..."):
            self._load_batch_files()
            
        if not self.all_sections:
            console.print("[red]No sections found in batch files[/red]")
            return
            
        console.print(f"[green]Loaded {len(self.all_sections)} sections[/green]\n")
        
        # Build hierarchical structure
        toc_sections = self._build_hierarchical_toc()
        
        # Build terms mapping
        self._build_terms_mapping()
        
        # Display statistics
        self._display_statistics()
        console.print()
        
        # Display document metadata if available
        if self.document_metadata:
            console.print("\n[bold cyan]Document Information[/bold cyan]")
            console.print(f"Title: {self.document_metadata.get('title', 'Unknown')}")
            console.print(f"Author: {self.document_metadata.get('author', 'Unknown')}")
            console.print(f"File: {self.document_metadata.get('file_path', 'Unknown')}")
        
        # Display TOC tree
        self._display_toc_tree(toc_sections)
        
        # Display terms summary if any
        if self.terms_mapping:
            self._display_terms_summary()
        
        # Save TOC
        if Confirm.ask("\nSave Table of Contents to file?", default=True):
            output_file = self._save_toc(toc_sections)
            console.print(f"\n[green]✓ Table of Contents saved to: {output_file}[/green]")
            
    async def run(self) -> None:
        """Main CLI loop."""
        console.print(
            Panel.fit(
                "[bold cyan]Table of Contents Generator[/bold cyan]\n"
                f"Output Directory: {self.output_dir}",
                title="TOC Generator",
            )
        )
        
        while True:
            console.print("\n[bold cyan]Options:[/bold cyan]")
            console.print("1. Generate ToC")
            console.print("0. Exit")
            
            choice = Prompt.ask("Select option", default="1")
            
            try:
                if choice == "1":
                    self._generate_toc()
                elif choice == "0":
                    break
                else:
                    console.print("[yellow]Invalid option[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if Confirm.ask("Show traceback?", default=False):
                    import traceback
                    traceback.print_exc()
                    

async def main():
    """Main entry point."""
    # Use default output directory
    output_dir = Path("output/technical_analysis_responses")
    
    # Allow optional override via command line
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
        
    if not output_dir.exists():
        console.print(f"[red]Output directory not found: {output_dir}[/red]")
        sys.exit(1)
        
    cli = TOCGeneratorCLI(output_dir)
    
    try:
        await cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    import anyio
    anyio.run(main)