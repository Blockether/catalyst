#!/usr/bin/env python3
"""
Interactive Pickle File Viewer and Query Tool

A powerful REPL for exploring pickle files with rich formatting,
query capabilities, intellisense, and interactive Python expressions.
"""

import pickle
import sys
import os
import argparse
import traceback
import time
from typing import Any, Optional, Dict, List
from pathlib import Path
import json
import pprint
import inspect
import keyword
import builtins
import re

from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.syntax import Syntax
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from rich.pretty import Pretty
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.rule import Rule
import ast
import glob
from pathlib import Path as PathlibPath
from difflib import SequenceMatcher

# Check for CLI dependencies since this is a tool
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.lexers import PygmentsLexer
    from prompt_toolkit.styles import Style
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.shortcuts import radiolist_dialog
    from prompt_toolkit.shortcuts import input_dialog
    from prompt_toolkit.application.current import get_app_session
    from prompt_toolkit.application import get_app
    from pygments.lexers.python import PythonLexer
except ImportError:
    print("Error: This tool requires CLI dependencies.")
    print("Install with: uv add 'com-blockether-catalyst[cli]'")
    print("or: pip install 'com-blockether-catalyst[cli]'")
    import sys
    sys.exit(1)


class PythonCompleter(Completer):
    """Custom completer for Python expressions with context awareness."""

    def __init__(self, locals_dict: Dict[str, Any]):
        """Initialize the completer with local variables."""
        self.locals_dict = locals_dict
        self.python_keywords = keyword.kwlist
        self.builtins_list = dir(builtins)

    def get_completions(self, document, complete_event):
        """Get completions for the current input."""
        text = document.text_before_cursor

        # Check if we're inside a string (for filepath completion)
        string_match = re.search(r'["\']([^"\']*)$', text)
        if string_match:
            partial_path = string_match.group(1)
            # quote_char = text[string_match.start()]  # Not used currently

            # Expand tilde to home directory
            if partial_path.startswith('~'):
                partial_path = os.path.expanduser(partial_path)

            # Get directory and file prefix
            if os.path.sep in partial_path:
                directory = os.path.dirname(partial_path)
                file_prefix = os.path.basename(partial_path)
            else:
                directory = '.'
                file_prefix = partial_path

            # Get completions for files/directories
            try:
                if os.path.exists(directory):
                    for item in os.listdir(directory):
                        if item.startswith(file_prefix):
                            full_path = os.path.join(directory, item)
                            # Add trailing slash for directories
                            if os.path.isdir(full_path):
                                item = item + os.path.sep
                                meta = 'directory'
                            else:
                                meta = 'file'

                            # For pickle files, show special indicator
                            if item.endswith(('.pkl', '.pickle', '.p')):
                                meta = 'pickle file'

                            yield Completion(
                                item,
                                start_position=-len(file_prefix),
                                display_meta=meta
                            )
            except (OSError, PermissionError):
                pass

            return

        # Handle empty input
        if not text:
            # Suggest main variables
            for key in self.locals_dict.keys():
                yield Completion(key, start_position=0, display_meta='variable')
            return

        # Split on dots to handle attribute access
        parts = text.split('.')

        # If we're completing after a dot (attribute access)
        if '.' in text and not text.endswith('.'):
            # Get the base object path and the partial attribute
            base_path = '.'.join(parts[:-1])
            partial_attr = parts[-1]

            try:
                # Evaluate the base object
                obj = eval(base_path, {}, self.locals_dict)

                # Get all attributes
                if hasattr(obj, '__dict__'):
                    attrs = dir(obj)
                else:
                    attrs = []

                # Add dict keys if it's a dict
                if isinstance(obj, dict):
                    attrs.extend(str(k) for k in obj.keys())

                # Filter and yield completions
                for attr in attrs:
                    if attr.startswith(partial_attr) and not attr.startswith('_'):
                        # Skip Pydantic internal attributes that cause warnings
                        if hasattr(type(obj), '__pydantic_fields_set__') and attr in ['model_fields', 'model_computed_fields', 'model_config']:
                            continue
                        yield Completion(
                            attr,
                            start_position=-len(partial_attr),
                            display_meta=self._get_attr_type(obj, attr)
                        )
            except Exception:
                pass

        # If we just typed a dot (want all attributes)
        elif text.endswith('.'):
            base_path = text[:-1]

            try:
                # Evaluate the base object
                obj = eval(base_path, {}, self.locals_dict)

                # Get all attributes
                if hasattr(obj, '__dict__'):
                    attrs = dir(obj)
                else:
                    attrs = []

                # Add dict keys if it's a dict
                if isinstance(obj, dict):
                    # For dict, show keys as completions
                    for key in obj.keys():
                        if isinstance(key, str):
                            yield Completion(
                                key,
                                start_position=0,
                                display_meta='key'
                            )
                        else:
                            yield Completion(
                                f'[{repr(key)}]',
                                start_position=0,
                                display_meta='key'
                            )

                # Show attributes
                for attr in attrs:
                    if not attr.startswith('_'):
                        # Skip Pydantic internal attributes that cause warnings
                        if hasattr(type(obj), '__pydantic_fields_set__') and attr in ['model_fields', 'model_computed_fields', 'model_config']:
                            continue
                        yield Completion(
                            attr,
                            start_position=0,
                            display_meta=self._get_attr_type(obj, attr)
                        )
            except Exception:
                pass

        # Completing variable names or keywords
        else:
            # Get the current partial word
            if ' ' in text:
                words = text.split()
                partial = words[-1]
                start_pos = -len(partial)
            else:
                partial = text
                start_pos = -len(text)

            # Complete variables
            for key in self.locals_dict.keys():
                if key.startswith(partial):
                    yield Completion(
                        key,
                        start_position=start_pos,
                        display_meta='variable'
                    )

            # Complete keywords
            for kw in self.python_keywords:
                if kw.startswith(partial):
                    yield Completion(
                        kw,
                        start_position=start_pos,
                        display_meta='keyword'
                    )

            # Complete builtins
            for builtin_name in self.builtins_list:
                if builtin_name.startswith(partial) and not builtin_name.startswith('_'):
                    yield Completion(
                        builtin_name,
                        start_position=start_pos,
                        display_meta='builtin'
                    )

    def _get_attr_type(self, obj: Any, attr: str) -> str:
        """Get a display type for an attribute."""
        try:
            # Check if it's a Pydantic model class attribute (to avoid deprecation warnings)
            if hasattr(type(obj), '__pydantic_fields_set__') and attr in ['model_fields', 'model_computed_fields', 'model_config']:
                return 'class_attribute'

            # Safely get the attribute value
            value = getattr(obj, attr)
            if callable(value):
                return 'method'
            elif isinstance(value, property):
                return 'property'
            else:
                return type(value).__name__
        except Exception:
            return 'attribute'


class PickleViewer:
    """Interactive pickle file viewer with query capabilities and intellisense."""

    MAX_DISPLAY_DEPTH = 5
    MAX_ITEMS_PER_LEVEL = 100
    DOUBLE_CTRL_C_TIMEOUT = 1.0  # seconds

    def __init__(self, filepath: Optional[Path] = None):
        """Initialize the pickle viewer."""
        self._console = Console()
        self._data: Any = None
        self._filepath: Optional[Path] = filepath
        self._history: List[str] = []
        self._locals: Dict[str, Any] = {}
        self._last_interrupt_time: float = 0
        self._pretty_print_mode: bool = False
        self._expand_path: Optional[str] = None

        # Setup prompt toolkit session
        self._setup_prompt_session()

        if filepath:
            self._load_file(filepath)

    def _setup_prompt_session(self):
        """Setup the prompt toolkit session with completions and styling."""
        # Create history file
        history_file = Path.home() / '.pickle_viewer_history'

        # Define custom style
        style = Style.from_dict({
            'prompt': '#00aa00 bold',
            'prompt.gt': '#00aa00 bold',
        })

        # Setup key bindings
        bindings = KeyBindings()

        @bindings.add('c-l')
        def _(event):
            """Ctrl+L: Open file chooser"""
            event.app.exit(result='__file_chooser__')

        @bindings.add('c-f')
        def _(event):
            """Ctrl+F: Fuzzy search in data"""
            event.app.exit(result='__fuzzy_search__')

        # Note: Ctrl+E is now handled within the interactive viewer, not here

        @bindings.add('escape')
        def _(event):
            """ESC: Cancel current input"""
            event.app.exit(result='')

        # Create the prompt session
        self._session = PromptSession(
            message=HTML('<prompt>&gt;&gt;&gt;</prompt> '),
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            enable_history_search=True,
            lexer=PygmentsLexer(PythonLexer),
            completer=None,  # Will be set after loading data
            complete_while_typing=True,
            style=style,
            include_default_pygments_style=True,
            mouse_support=True,
            complete_in_thread=True,
            key_bindings=bindings,
        )

    def _update_completer(self):
        """Update the completer with current locals."""
        self._session.completer = PythonCompleter(self._locals)

    def _open_file_chooser(self) -> None:
        """Open a file chooser dialog to select a pickle file."""
        try:
            # Find pickle files in current directory and subdirectories
            pickle_files = []

            # Search for common pickle file patterns
            patterns = ['*.pkl', '*.pickle', '*.p']
            for pattern in patterns:
                # Current directory
                pickle_files.extend(glob.glob(pattern))
                # Subdirectories (one level deep)
                pickle_files.extend(glob.glob(f'*/{pattern}'))
                pickle_files.extend(glob.glob(f'**/{pattern}', recursive=True))

            # Remove duplicates and sort
            pickle_files = sorted(list(set(pickle_files)))

            if not pickle_files:
                self._console.print("[yellow]No pickle files found in current directory[/yellow]")
                # Allow manual path entry
                try:
                    path = Prompt.ask("Enter pickle file path (ESC to cancel)")
                    if path:
                        self._load_file(Path(path))
                        self.display_overview()
                except (KeyboardInterrupt, EOFError):
                    pass
                return

            # Show options in console
            self._console.print("\n[bold cyan]Available Pickle Files:[/bold cyan]")
            for i, file in enumerate(pickle_files[:20], 1):
                self._console.print(f"  {i:2d}. {file}")

            if len(pickle_files) > 20:
                self._console.print(f"  ... and {len(pickle_files) - 20} more files")

            # Get user selection
            try:
                choice = Prompt.ask(
                    "Enter file number, full path, or ESC/Enter to cancel",
                    default=""
                )

                if not choice:  # ESC or Enter pressed
                    return
            except (KeyboardInterrupt, EOFError):
                return

            if choice:
                try:
                    # Check if it's a number
                    idx = int(choice) - 1
                    if 0 <= idx < len(pickle_files[:20]):
                        self._load_file(Path(pickle_files[idx]))
                        self.display_overview()
                except ValueError:
                    # Treat as path
                    self._load_file(Path(choice))
                    self.display_overview()

        except Exception as e:
            self._console.print(f"[red]Error loading file:[/red] {e}")

    def _interactive_expand(self) -> None:
        """Interactive navigation and expansion of data structures."""
        if not self._data:
            self._console.print("[yellow]No data loaded[/yellow]")
            return

        # Start numbered navigation with the main data
        self._numbered_navigation(self._data, "x")

    def _numbered_navigation(self, start_obj: Any, start_path: str) -> None:
        """Numbered navigation mode for data structures."""
        current_obj = start_obj
        current_path = start_path
        navigation_stack = [(start_obj, start_path)]  # Stack for back navigation

        while True:
            # Clear screen and show current location
            self._console.clear()
            self._console.print(Rule(f"[bold cyan]Data Navigator - Path: {current_path}[/bold cyan]"))

            # Get available keys/indices
            options = []

            if isinstance(current_obj, dict):
                for key in list(current_obj.keys())[:50]:  # Limit to 50 items
                    options.append((str(key), key, f"[{repr(key)}]"))
            elif isinstance(current_obj, (list, tuple)):
                for i in range(min(len(current_obj), 50)):
                    item = current_obj[i]
                    preview = str(item)[:50] + "..." if len(str(item)) > 50 else str(item)
                    options.append((str(i), i, f"[{i}]: {preview}"))
            elif hasattr(current_obj, '__dict__'):
                # For Pydantic models, use model_dump() if available
                if hasattr(current_obj, 'model_dump'):
                    try:
                        attrs = current_obj.model_dump()
                    except Exception:
                        attrs = vars(current_obj)
                else:
                    attrs = vars(current_obj)

                for attr_name in list(attrs.keys())[:50]:
                    if not attr_name.startswith('_'):
                        options.append((attr_name, attr_name, f".{attr_name}"))

            if not options:
                # No expandable items - show the value
                tree = Tree(f"[bold]{type(current_obj).__name__}[/bold]")
                if isinstance(current_obj, (str, int, float, bool, type(None))):
                    tree.add(self._format_value(current_obj))
                else:
                    self._build_tree(current_obj, tree, depth=0)
                self._console.print(tree)
                self._console.print("\n[yellow]No expandable items. Press Enter to go back.[/yellow]")
                Prompt.ask("")
                if len(navigation_stack) > 1:
                    navigation_stack.pop()
                    current_obj, current_path = navigation_stack[-1]
                else:
                    break
                continue

            # Show current object overview
            tree = Tree(f"[bold]{type(current_obj).__name__}[/bold]")
            self._build_tree(current_obj, tree, depth=1)  # Show only 1 level
            self._console.print(tree)

            # Show navigation options
            self._console.print("\n[bold cyan]Numbered Navigation:[/bold cyan]")
            self._console.print("[bold]Enter number to expand, 'b' to go back, 'q' to quit:[/bold]\n")

            # Display numbered options
            for i, (display, key, path_str) in enumerate(options, 1):
                if isinstance(current_obj, dict):
                    value = current_obj[key]
                elif isinstance(current_obj, (list, tuple)):
                    value = current_obj[key]
                else:
                    # For objects with attributes, check if it's a Pydantic model
                    if hasattr(current_obj, 'model_dump'):
                        # Use model_dump to get the value safely
                        model_data = current_obj.model_dump()
                        value = model_data.get(key, getattr(current_obj, key, None))
                    else:
                        value = getattr(current_obj, key)
                type_str = type(value).__name__
                self._console.print(f"  {i:2d}. {path_str} [[bold magenta]{type_str}[/bold magenta]]")

            # Get user choice
            try:
                choice = Prompt.ask("Choice", default="q")

                if choice.lower() == 'q':
                    break
                elif choice.lower() == 'b':
                    if len(navigation_stack) > 1:
                        navigation_stack.pop()
                        current_obj, current_path = navigation_stack[-1]
                    continue

                # Try to parse as number
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(options):
                        display, key, path_str = options[idx]
                        # Navigate to selected item
                        if isinstance(current_obj, dict):
                            new_obj = current_obj[key]
                            new_path = f"{current_path}[{repr(key)}]"
                        elif isinstance(current_obj, (list, tuple)):
                            new_obj = current_obj[key]
                            new_path = f"{current_path}[{key}]"
                        else:
                            # For objects with attributes, check if it's a Pydantic model
                            if hasattr(current_obj, 'model_dump'):
                                # Use model_dump to get the value safely
                                model_data = current_obj.model_dump()
                                new_obj = model_data.get(key, getattr(current_obj, key, None))
                            else:
                                new_obj = getattr(current_obj, key)
                            new_path = f"{current_path}.{key}"

                        current_obj = new_obj
                        current_path = new_path
                        navigation_stack.append((current_obj, current_path))
                except ValueError:
                    self._console.print("[red]Invalid choice[/red]")
                    time.sleep(1)

            except (KeyboardInterrupt, EOFError):
                break

        # Clear screen when done
        self._console.clear()

    def _interactive_result_view(self, result: Any) -> None:
        """Display result with interactive expand/collapse navigation."""
        from prompt_toolkit import prompt
        from prompt_toolkit.key_binding import KeyBindings

        # Track expanded paths
        expanded_paths = set()
        expanded_paths.add('')  # Root is always expanded
        current_selection = 0
        viewport_top = 0  # Track the top of the viewport

        # Terminal size management
        try:
            terminal_height = self._console.height - 8  # Leave room for header and instructions
        except Exception:
            terminal_height = 20  # Default fallback

        # Build a flat list of all visible items with their paths
        def get_visible_items(obj, path='', depth=0):
            items = []
            if path in expanded_paths or path == '':
                if isinstance(obj, dict):
                    for key in list(obj.keys())[:50]:
                        item_path = f"{path}[{repr(key)}]" if path else f"[{repr(key)}]"
                        value = obj[key]
                        has_children = isinstance(value, (dict, list, tuple)) and len(value) > 0
                        items.append((depth, key, value, item_path, has_children))
                        if item_path in expanded_paths:
                            items.extend(get_visible_items(value, item_path, depth + 1))
                elif isinstance(obj, (list, tuple)):
                    for i in range(min(len(obj), 50)):
                        item_path = f"{path}[{i}]" if path else f"[{i}]"
                        value = obj[i]
                        has_children = isinstance(value, (dict, list, tuple)) and len(value) > 0
                        items.append((depth, i, value, item_path, has_children))
                        if item_path in expanded_paths:
                            items.extend(get_visible_items(value, item_path, depth + 1))
                elif hasattr(obj, '__dict__'):
                    # For Pydantic models, use model_dump() if available
                    if hasattr(obj, 'model_dump'):
                        try:
                            attrs = obj.model_dump()
                        except Exception:
                            attrs = {k: v for k, v in vars(obj).items() if not k.startswith('_')}
                    else:
                        attrs = vars(obj)

                    for key in list(attrs.keys())[:50]:
                        if not str(key).startswith('_'):
                            item_path = f"{path}.{key}" if path else f".{key}"
                            value = attrs[key]
                            has_children = isinstance(value, (dict, list, tuple)) and len(value) > 0
                            items.append((depth, key, value, item_path, has_children))
                            if item_path in expanded_paths:
                                items.extend(get_visible_items(value, item_path, depth + 1))
            return items

        def render_display():
            """Render the current display state."""
            nonlocal viewport_top

            # Get visible items
            items = get_visible_items(result)

            # Adjust viewport to keep selection visible
            if current_selection < viewport_top:
                viewport_top = current_selection
            elif current_selection >= viewport_top + terminal_height:
                viewport_top = current_selection - terminal_height + 1

            # Clear screen and redraw
            self._console.clear()
            self._console.print(Rule("[bold cyan]Interactive Result Viewer[/bold cyan]"))
            self._console.print("[bold]Keys:[/bold] ↑/↓ or j/k: navigate | Enter/Space: expand/collapse | Ctrl+E: numbered mode | q: quit\n")

            # Display items within viewport
            viewport_bottom = min(viewport_top + terminal_height, len(items))
            for idx in range(viewport_top, viewport_bottom):
                if idx >= len(items):
                    break

                depth, key, value, path, has_children = items[idx]
                indent = "  " * depth

                # Format the display
                if has_children:
                    if path in expanded_paths:
                        marker = "[bold red][-][/bold red]"
                    else:
                        marker = "[bold green][+][/bold green]"
                else:
                    marker = "   "

                # Format key and value
                if isinstance(key, int):
                    key_str = f"[{key}]"
                else:
                    key_str = str(key)

                type_str = f"[bold magenta]{type(value).__name__}[/bold magenta]"

                # Build the line
                if isinstance(value, (str, int, float, bool, type(None))):
                    value_preview = self._format_value(value)
                    line = f"{indent}{marker} {key_str}: {value_preview}"
                else:
                    line = f"{indent}{marker} {key_str}: {type_str}"

                # Highlight current selection
                if idx == current_selection:
                    line = f"[reverse]{line}[/reverse]"

                self._console.print(line, highlight=False)

            # Show position indicator if there are more items
            if len(items) > terminal_height:
                position = f"[{current_selection + 1}/{len(items)}]"
                self._console.print(f"\n[dim]{position}[/dim]")

            return items

        # Initial display
        items = render_display()

        # Main loop
        while True:
            try:
                # Setup key bindings
                kb = KeyBindings()
                result_key: List[Optional[str]] = [None]

                @kb.add('up')
                def _(event):
                    result_key[0] = 'up'
                    event.app.exit()

                @kb.add('k')
                def _(event):
                    result_key[0] = 'up'
                    event.app.exit()

                @kb.add('down')
                def _(event):
                    result_key[0] = 'down'
                    event.app.exit()

                @kb.add('j')
                def _(event):
                    result_key[0] = 'down'
                    event.app.exit()

                @kb.add('enter')
                @kb.add('space')
                def _(event):
                    result_key[0] = 'toggle'
                    event.app.exit()

                @kb.add('q')
                @kb.add('escape')
                def _(event):
                    result_key[0] = 'quit'
                    event.app.exit()

                @kb.add('c-e')
                def _(event):
                    result_key[0] = 'numbered_mode'
                    event.app.exit()

                # Wait for input
                prompt('', key_bindings=kb)

                # Process action
                if result_key[0] == 'quit':
                    break
                elif result_key[0] == 'numbered_mode':
                    # Switch to numbered navigation mode
                    if current_selection < len(items):
                        _, _, selected_value, selected_path, _ = items[current_selection]
                        self._numbered_navigation(selected_value, selected_path)
                        items = render_display()  # Redraw after returning
                elif result_key[0] == 'up':
                    if current_selection > 0:
                        current_selection -= 1
                        items = render_display()
                elif result_key[0] == 'down':
                    if current_selection < len(items) - 1:
                        current_selection += 1
                        items = render_display()
                elif result_key[0] == 'toggle':
                    if current_selection < len(items):
                        _, _, _, path, has_children = items[current_selection]
                        if has_children:
                            if path in expanded_paths:
                                # Collapse this and all children
                                expanded_paths.discard(path)
                                # Remove all child paths
                                to_remove = [p for p in expanded_paths if p.startswith(path)]
                                for p in to_remove:
                                    expanded_paths.discard(p)
                            else:
                                expanded_paths.add(path)
                            items = render_display()

            except (KeyboardInterrupt, EOFError):
                break

        # Clear screen when done
        self._console.clear()

    def _fuzzy_search(self) -> None:
        """Perform fuzzy search in data structures."""
        if not self._locals:
            self._console.print("[yellow]No variables defined[/yellow]")
            return

        # Show available variables
        var_list = [k for k in self._locals.keys() if not k.startswith('_') and k not in
                    ['len', 'type', 'dir', 'vars', 'isinstance', 'hasattr', 'getattr',
                     'filter', 'map', 'list', 'dict', 'set', 'tuple', 'sum', 'min', 'max',
                     'sorted', 'reversed', 'enumerate', 'zip', 'any', 'all', 'print']]

        if var_list:
            self._console.print("[bold cyan]Available variables:[/bold cyan] " + ", ".join(f"[bold green]{v}[/bold green]" for v in var_list))

        # Create temporary prompt sessions with proper ESC handling
        kb = KeyBindings()

        @kb.add('escape', eager=True)
        def _(event):
            event.app.exit(result='')

        @kb.add('c-c')
        def _(event):
            event.app.exit(result='')

        # Get the variable to search in
        try:
            var_session = PromptSession(
                message="Enter variable to search in (ESC to cancel): ",
                key_bindings=kb,
                enable_open_in_editor=False
            )
            var_name = var_session.prompt(default=var_list[0] if var_list else 'x')
            if not var_name:  # ESC pressed
                return
        except (KeyboardInterrupt, EOFError):
            return

        if var_name not in self._locals:
            self._console.print(f"[red]Variable '{var_name}' not found[/red]")
            return

        # Get search pattern
        try:
            pattern_session = PromptSession(
                message="Enter search pattern (fuzzy matching, ESC to cancel): ",
                key_bindings=kb,
                enable_open_in_editor=False
            )
            pattern = pattern_session.prompt()
            if not pattern:
                return
        except (KeyboardInterrupt, EOFError):
            return

        # Get the object to search
        obj = self._locals[var_name]

        # Perform fuzzy search
        results = self._fuzzy_search_recursive(obj, pattern, path=var_name)

        if results:
            self._console.print(f"\n[bold green]Found {len(results)} matches:[/bold green]")
            for path, value, score in results[:20]:  # Limit to 20 results
                self._console.print(f"  [bold blue]{path}[/bold blue] (score: {score:.2f})")
                self._console.print(f"    → {self._format_value(value, max_length=60)}")

            if len(results) > 20:
                self._console.print(f"\n[bold yellow]... and {len(results) - 20} more matches[/bold yellow]")
        else:
            self._console.print(f"[yellow]No matches found for '{pattern}'[/yellow]")

    def _fuzzy_search_recursive(self, obj: Any, pattern: str, path: str = '',
                                 results: Optional[List] = None, depth: int = 0) -> List:
        """Recursively search for fuzzy matches in data structure."""
        if results is None:
            results = []

        if depth > 5:  # Limit depth to prevent infinite recursion
            return results

        pattern_lower = pattern.lower()

        # Helper function to calculate fuzzy match score
        def fuzzy_score(text: str) -> float:
            text_lower = str(text).lower()
            # Check for substring match first (higher score)
            if pattern_lower in text_lower:
                return 1.0
            # Use SequenceMatcher for fuzzy matching
            return SequenceMatcher(None, pattern_lower, text_lower).ratio()

        try:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Check if key matches
                    key_score = fuzzy_score(str(key))
                    if key_score > 0.6:  # Threshold for fuzzy match
                        results.append((f"{path}[{repr(key)}]", value, key_score))

                    # Check if string value matches
                    if isinstance(value, str):
                        value_score = fuzzy_score(value)
                        if value_score > 0.6:
                            results.append((f"{path}[{repr(key)}]", value, value_score))

                    # Recurse into nested structures
                    if isinstance(value, (dict, list, tuple)):
                        self._fuzzy_search_recursive(value, pattern, f"{path}[{repr(key)}]", results, depth + 1)

            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj[:100]):  # Limit to first 100 items
                    # Check if string item matches
                    if isinstance(item, str):
                        item_score = fuzzy_score(item)
                        if item_score > 0.6:
                            results.append((f"{path}[{i}]", item, item_score))

                    # Recurse into nested structures
                    if isinstance(item, (dict, list, tuple)):
                        self._fuzzy_search_recursive(item, pattern, f"{path}[{i}]", results, depth + 1)

            elif hasattr(obj, '__dict__'):
                for attr, value in vars(obj).items():
                    # Check attribute name
                    attr_score = fuzzy_score(attr)
                    if attr_score > 0.6:
                        results.append((f"{path}.{attr}", value, attr_score))

                    # Check if string value matches
                    if isinstance(value, str):
                        value_score = fuzzy_score(value)
                        if value_score > 0.6:
                            results.append((f"{path}.{attr}", value, value_score))

                    # Recurse into nested structures
                    if isinstance(value, (dict, list, tuple)):
                        self._fuzzy_search_recursive(value, pattern, f"{path}.{attr}", results, depth + 1)

        except Exception:
            pass  # Skip items that can't be accessed

        # Sort by score (highest first) and return
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def _load_file(self, filepath: Path) -> None:
        """Load a pickle file."""
        try:
            with open(filepath, 'rb') as f:
                self._data = pickle.load(f)
            self._filepath = filepath
            self._locals = {'x': self._data}
            self._update_completer()
            self._console.print(f"[green]✓[/green] Loaded: {filepath}")
            self._console.print(f"[dim]Type: {type(self._data).__name__}[/dim]")
        except Exception as e:
            self._console.print(f"[red]✗[/red] Error loading file: {e}")
            raise

    def _build_tree(self, obj: Any, tree: Tree, depth: int = 0, max_depth: Optional[int] = None) -> None:
        """Recursively build a rich tree from an object."""
        if max_depth is None:
            max_depth = self.MAX_DISPLAY_DEPTH

        if depth >= max_depth:
            if depth > 0:  # Only show this for nested items
                tree.add("[bold green][+][/bold green] [bold yellow](expand with Ctrl+E)[/bold yellow]")
            return

        if isinstance(obj, dict):
            items = list(obj.items())[:self.MAX_ITEMS_PER_LEVEL]
            for key, value in items:
                key_str = f"[bold cyan]{repr(key)}[/bold cyan]"
                value_type = f"[bold magenta]{type(value).__name__}[/bold magenta]"

                if isinstance(value, (dict, list, tuple, set)):
                    branch = tree.add(f"{key_str}: {value_type}")
                    self._build_tree(value, branch, depth + 1, max_depth)
                else:
                    value_str = self._format_value(value)
                    tree.add(f"{key_str}: {value_str}")

            if len(obj) > self.MAX_ITEMS_PER_LEVEL:
                tree.add(f"[bold yellow]... and {len(obj) - self.MAX_ITEMS_PER_LEVEL} more items[/bold yellow]")

        elif isinstance(obj, (list, tuple, set)):
            items = list(obj)[:self.MAX_ITEMS_PER_LEVEL]
            for i, item in enumerate(items):
                item_type = f"[bold magenta]{type(item).__name__}[/bold magenta]"

                if isinstance(item, (dict, list, tuple, set)):
                    branch = tree.add(f"[{i}]: {item_type}")
                    self._build_tree(item, branch, depth + 1, max_depth)
                else:
                    value_str = self._format_value(item)
                    tree.add(f"[{i}]: {value_str}")

            if len(obj) > self.MAX_ITEMS_PER_LEVEL:
                tree.add(f"[bold yellow]... and {len(obj) - self.MAX_ITEMS_PER_LEVEL} more items[/bold yellow]")

        elif hasattr(obj, '__dict__'):
            # For Pydantic models, use model_dump() if available
            if hasattr(obj, 'model_dump'):
                try:
                    attrs = obj.model_dump()
                except Exception:
                    # If model_dump fails, try to get non-deprecated attributes
                    attrs = {k: v for k, v in vars(obj).items() if not k.startswith('_')}
            else:
                attrs = vars(obj)

            for key, value in list(attrs.items())[:self.MAX_ITEMS_PER_LEVEL]:
                key_str = f"[bold magenta]{key}[/bold magenta]"
                value_type = f"[bold cyan]{type(value).__name__}[/bold cyan]"

                if isinstance(value, (dict, list, tuple, set)):
                    branch = tree.add(f"{key_str}: {value_type}")
                    self._build_tree(value, branch, depth + 1, max_depth)
                else:
                    value_str = self._format_value(value)
                    tree.add(f"{key_str}: {value_str}")
        else:
            tree.add(self._format_value(obj))

    def _format_value(self, value: Any, max_length: int = 80) -> str:
        """Format a value for display with colors visible in both dark and light themes."""
        if value is None:
            return "[bold magenta]None[/bold magenta]"
        elif isinstance(value, bool):
            return f"[bold cyan]{value}[/bold cyan]"
        elif isinstance(value, (int, float)):
            return f"[bold blue]{value}[/bold blue]"
        elif isinstance(value, str):
            if len(value) > max_length:
                value = value[:max_length] + "..."
            return f"[bold green]{repr(value)}[/bold green]"
        elif isinstance(value, bytes):
            return f"[bold red]<bytes: {len(value)} bytes>[/bold red]"
        else:
            str_repr = str(value)
            if len(str_repr) > max_length:
                str_repr = str_repr[:max_length] + "..."
            return f"[bold yellow]{str_repr}[/bold yellow]"

    def display_overview(self) -> None:
        """Display an overview of the loaded data."""
        if self._data is None:
            self._console.print("[yellow]No data loaded[/yellow]")
            return

        # Create overview panel
        overview_content = []
        overview_content.append(f"[bold]File:[/bold] {self._filepath}")
        overview_content.append(f"[bold]Type:[/bold] {type(self._data).__name__}")

        if isinstance(self._data, (list, tuple, set, dict)):
            overview_content.append(f"[bold]Length:[/bold] {len(self._data)}")

        if sys.getsizeof(self._data, 0) < 1024 * 1024:
            size_str = f"{sys.getsizeof(self._data, 0):,} bytes"
        else:
            size_str = f"{sys.getsizeof(self._data, 0) / 1024 / 1024:.2f} MB"
        overview_content.append(f"[bold]Size in memory:[/bold] {size_str}")
        overview_content.append("")
        overview_content.append("[bold cyan]Available variable:[/bold cyan] [bold green]x[/bold green] (your loaded data)")
        overview_content.append("[bold]Use 'x' to access your data in queries[/bold]")
        overview_content.append("")
        overview_content.append("[bold yellow]Intellisense:[/bold yellow] Press [cyan]Tab[/cyan] for autocomplete")

        panel = Panel(
            "\n".join(overview_content),
            title="[bold blue]Data Overview[/bold blue]",
            border_style="blue"
        )
        self._console.print(panel)

        # Display with numbered navigation (the good mode!)
        self._console.print("\n[bold]Data Structure:[/bold]")
        self._numbered_navigation(self._data, "x")

    def execute_query(self, query: str) -> Any:
        """Execute a Python expression or statement as a query on the data."""
        try:
            # Update locals with common utilities
            self._locals.update({
                'x': self._data,
                'len': len,
                'type': type,
                'dir': dir,
                'vars': vars,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'filter': filter,
                'map': map,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'sum': sum,
                'min': min,
                'max': max,
                'sorted': sorted,
                'reversed': reversed,
                'enumerate': enumerate,
                'zip': zip,
                'any': any,
                'all': all,
                'print': rprint,
                # Add import capabilities
                '__import__': __import__,
                '__builtins__': __builtins__,
            })

            # Update completer with new locals
            self._update_completer()

            # Try to parse the query
            try:
                parsed = ast.parse(query, mode='single')

                # Check if it's an import statement
                if isinstance(parsed.body[0], (ast.Import, ast.ImportFrom)):
                    exec(query, self._locals, self._locals)
                    self._console.print(f"[green]✓[/green] {query}")
                    self._update_completer()
                    return None

                # Check if it's an assignment
                elif isinstance(parsed.body[0], ast.Assign):
                    exec(query, self._locals, self._locals)
                    target = parsed.body[0].targets[0]
                    if isinstance(target, ast.Name):
                        var_name = target.id
                    else:
                        var_name = str(target)
                    result = self._locals.get(var_name)
                    self._console.print(f"[green]✓[/green] {var_name} = {self._format_value(result)}")
                    self._update_completer()
                    return result

                # Check if it's a function/class definition or other statement
                elif isinstance(parsed.body[0], (ast.FunctionDef, ast.ClassDef, ast.For, ast.While, ast.With, ast.If)):
                    exec(query, self._locals, self._locals)
                    self._console.print("[green]✓[/green] Executed")
                    self._update_completer()
                    return None

                # Otherwise try to evaluate as expression
                else:
                    result = eval(query, self._locals, self._locals)
                    return result

            except SyntaxError:
                # If parsing fails, try as expression
                result = eval(query, self._locals, self._locals)
                return result

        except Exception as e:
            self._console.print(f"[red]Error:[/red] {e}")
            return None

    def display_result(self, result: Any) -> None:
        """Display a query result with numbered navigation."""
        if result is None:
            return

        # For simple types, just display them
        if isinstance(result, (str, int, float, bool, type(None))):
            tree = Tree(f"[bold]{type(result).__name__}[/bold]")
            tree.add(self._format_value(result))
            self._console.print(tree)
            return

        # Go straight to numbered navigation mode (the good one!)
        self._numbered_navigation(result, "x")


    def show_help(self) -> None:
        """Display help information."""
        help_text = """
[bold cyan]Pickle Viewer Commands:[/bold cyan]

[bold]Basic Commands:[/bold]
  [cyan]help[/cyan]     - Show this help message
  [cyan]clear[/cyan]    - Clear the screen
  [cyan]history[/cyan]  - Show query history
  [cyan]exit[/cyan]     - Exit the viewer

[bold]Hotkeys:[/bold]
  [cyan]Ctrl+L[/cyan]   - Open pickle file chooser dialog
  [cyan]Ctrl+F[/cyan]   - Fuzzy search in data structures

[bold]Intellisense Features:[/bold]
  [cyan]Tab[/cyan]      - Autocomplete variables, attributes, and methods
  [cyan]↑/↓[/cyan]      - Navigate through history
  [cyan]Ctrl+R[/cyan]   - Search history

[bold]Query Examples:[/bold]
  [cyan]x[/cyan]                        - Show the entire data
  [cyan]x.[Tab][/cyan]                  - Show all attributes/methods
  [cyan]type(x)[/cyan]                  - Get data type
  [cyan]len(x)[/cyan]                   - Get data length
  [cyan]x[0][/cyan]                     - Access first element
  [cyan]x['key'][/cyan]                 - Access dictionary key
  [cyan]x.attribute[/cyan]              - Access object attribute
  [cyan][item for item in x if ...][/cyan] - List comprehension
  [cyan]result = x[0:10][/cyan]         - Store result in variable

[bold]Available Variables:[/bold]
  [cyan]x[/cyan] - The loaded pickle data (your main data object)
  [cyan]Any variables you create are preserved[/cyan]

[bold]Tips:[/bold]
  • Press Tab for autocomplete at any time
  • Use dot notation to explore object attributes
  • History is saved between sessions
  • Results are automatically formatted with colors
  • Large structures are truncated for readability
  • Press Ctrl+C to interrupt long operations
  • Press Ctrl+C twice quickly to exit
  • Press Ctrl+D to exit
        """
        self._console.print(Panel(help_text, title="[bold]Help[/bold]", border_style="green"))

    def run_repl(self) -> None:
        """Run the interactive REPL with intellisense."""
        # Display welcome screen with all features
        self._console.clear()
        self._console.print(Rule("[bold cyan]🔍 Pickle Viewer - Interactive REPL with Intellisense[/bold cyan]"))

        # Display feature panels
        welcome_info = Table.grid(padding=1)
        welcome_info.add_column(style="cyan", justify="left")
        welcome_info.add_column(style="white", justify="left")

        # Quick start
        welcome_info.add_row("[bold yellow]Quick Start:[/bold yellow]", "")
        welcome_info.add_row("  Ctrl+L", "Load pickle file (file chooser)")
        welcome_info.add_row("  Ctrl+E", "Switch to numbered mode (in interactive view)")
        welcome_info.add_row("  Tab", "Autocomplete variables, attributes, paths")
        welcome_info.add_row("  x", "Your loaded data (main variable)")
        welcome_info.add_row("", "")

        # Essential hotkeys
        welcome_info.add_row("[bold yellow]Essential Hotkeys:[/bold yellow]", "")
        welcome_info.add_row("  Ctrl+F", "Fuzzy search in data structures")
        welcome_info.add_row("  ESC", "Cancel current prompt/dialog")
        welcome_info.add_row("  Ctrl+C (2x)", "Exit the viewer")
        welcome_info.add_row("  Ctrl+D", "Exit the viewer")
        welcome_info.add_row("", "")

        # Commands
        welcome_info.add_row("[bold yellow]Commands:[/bold yellow]", "")
        welcome_info.add_row("  help", "Show detailed help")
        welcome_info.add_row("  clear", "Clear screen")
        welcome_info.add_row("  history", "Show command history")
        welcome_info.add_row("  exit/quit", "Exit viewer")

        panel = Panel(
            welcome_info,
            title="[bold green]Welcome to Pickle Viewer[/bold green]",
            border_style="green",
            expand=True
        )
        self._console.print(panel)

        # Python capabilities
        python_info = [
            "[bold cyan]Python Capabilities:[/bold cyan]",
            "  • Full Python expressions and statements",
            "  • Import modules: [green]import os, json, numpy[/green]",
            "  • Define functions and classes",
            "  • List comprehensions: [green][x for x in data if ...][/green]",
            "  • Store results: [green]result = x['key'][0][/green]",
            "",
            "[bold cyan]Autocomplete Features:[/bold cyan]",
            "  • File paths in strings: [green]\"~/file.pkl\"[/green] + Tab",
            "  • Object attributes: [green]x.[/green] + Tab",
            "  • Dict keys: [green]x['[/green] + Tab",
            "  • Nested paths: [green]x[0].attribute.[/green] + Tab",
        ]

        info_panel = Panel(
            "\n".join(python_info),
            title="[bold blue]Features[/bold blue]",
            border_style="blue",
            expand=True
        )
        self._console.print(info_panel)

        self._console.print("\n[bold]Type 'help' for detailed documentation[/bold]")
        self._console.print(Rule(style="bold blue"))

        if self._data is not None:
            self.display_overview()
        else:
            self._console.print("\n[yellow]No file loaded. Press Ctrl+L to load a pickle file.[/yellow]\n")

        while True:
            try:
                # Get input with autocomplete
                query = self._session.prompt()

                # Handle special exit codes from key bindings
                if query == '__file_chooser__':
                    self._open_file_chooser()
                    continue
                elif query == '__fuzzy_search__':
                    self._fuzzy_search()
                    continue

                if not query or not query.strip():
                    continue

                # Store in history
                self._history.append(query)

                # Handle special commands
                if query.lower() in ['exit', 'quit', 'q']:
                    if Confirm.ask("Exit viewer?"):
                        break

                elif query.lower() in ['help', 'h', '?']:
                    self.show_help()

                elif query.lower() == 'clear':
                    self._console.clear()

                elif query.lower() == 'history':
                    self._console.print("[bold]Query History:[/bold]")
                    for i, cmd in enumerate(self._history[-20:], 1):
                        self._console.print(f"  {i:3d}. {cmd}")

                else:
                    # Execute as Python query
                    result = self.execute_query(query)
                    self.display_result(result)

            except KeyboardInterrupt:
                current_time = time.time()
                if current_time - self._last_interrupt_time < self.DOUBLE_CTRL_C_TIMEOUT:
                    self._console.print("\n[yellow]Double Ctrl+C detected. Exiting...[/yellow]")
                    break
                else:
                    self._console.print("\n[yellow]Interrupted (press Ctrl+C again quickly to exit)[/yellow]")
                    self._last_interrupt_time = current_time
                continue
            except EOFError:
                break
            except Exception as e:
                self._console.print(f"[red]Error:[/red] {e}")
                if '--debug' in sys.argv:
                    self._console.print(traceback.format_exc())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Pickle File Viewer and Query Tool with Intellisense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.pkl           # Open a pickle file
  %(prog)s                    # Start REPL without loading a file
  %(prog)s data.pkl --debug   # Enable debug mode

Usage:
  uv run python3 tools/PickleViewer.py yourfile.pkl

In the REPL:
  - Your data is available as 'x'
  - Press Tab for autocomplete
  - Use arrow keys to navigate history
  - Ctrl+R to search history
        """
    )

    parser.add_argument(
        'file',
        nargs='?',
        help='Pickle file to load'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with full tracebacks'
    )

    args = parser.parse_args()

    try:
        if args.file:
            filepath = Path(args.file)
            if not filepath.exists():
                print(f"Error: File '{filepath}' not found")
                sys.exit(1)
            viewer = PickleViewer(filepath)
        else:
            viewer = PickleViewer()
            viewer._console.print("[yellow]No file loaded. Use Ctrl+L to load a pickle file.[/yellow]")

        viewer.run_repl()

    except Exception as e:
        print(f"Fatal error: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
