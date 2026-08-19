"""Static notebook-to-repository call audit for cleanup planning.

This script does not execute notebooks or analysis code. It parses Python
source and notebook code cells, then reports which repository-defined
functions/methods are reachable from the selected notebooks.
"""

from __future__ import annotations

import ast
import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path


# Notebooks treated as the current analysis pipeline entry points.
NOTEBOOKS = (
    "WT-LP_and_ConGeo.ipynb",
    "OptogeneticFigure.ipynb",
    "evaluation.ipynb",
)


# WT/opto notebooks seed recursive search only from these analysis objects.
PLOT_NOTEBOOK_ROOT_OBJECTS = {"plotter", "stats_runner"}


# Evaluation notebook seeds recursive search only from these explicitly imported functions.
EVALUATION_ROOT_FUNCTIONS = {
    "plot_2d_keypoint_xy_traces",
    "plot_tracking_qc_summary",
    "summarize_tracking_qc_by_trial_keypoint",
}


# Folders intentionally skipped so the audit never scans data or generated output.
EXCLUDED_DIRS = {
    ".git",
    ".ipynb_checkpoints",
    "__pycache__",
    "KinematicData",
    "Metadata",
    "2DprojectionData",
    "2D projectionData",
    "Figures",
}


# This script is a reporting tool, not part of the analysis pipeline.
SELF_FILE = "repo_cleaner.py"


# Builtin/common constructor names are ignored when reporting unresolved calls.
IGNORED_CALL_NAMES = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "display",
    "enumerate",
    "FileNotFoundError",
    "float",
    "getattr",
    "int",
    "isinstance",
    "iter",
    "join",
    "KeyError",
    "len",
    "list",
    "max",
    "mean",
    "min",
    "next",
    "Path",
    "print",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "ValueError",
    "zip",
}


# Attribute method names that are ordinary containers/data objects, not repo methods.
IGNORED_METHOD_NAMES = {
    "append",
    "items",
}


# Variable names with stable repository class meanings across the analysis code.
KNOWN_VARIABLE_CLASSES = {
    "group_info": "Group",
    "current_group": "Group",
    "group_a": "Group",
    "group_b": "Group",
    "trial_info": "Trial",
    "calculator": "SimpleCalculation",
    "qc_config": "TrackingQCConfig",
}


@dataclass
class FunctionRecord:
    """Repository-defined function or method plus its local static context."""

    qualified_name: str
    module: str
    file: Path
    line: int
    class_name: str | None = None
    node: ast.AST | None = None
    import_aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class CellRecord:
    """Notebook cell calls resolved to repository-defined symbols."""

    notebook: str
    cell_number: int
    direct_calls: set[str] = field(default_factory=set)
    unresolved: set[str] = field(default_factory=set)


class RepoCallAnalyzer:
    """Build and print a static call graph rooted in notebook code cells."""

    def __init__(self, root: Path):
        # Resolve the workspace once so all reported paths can be relative.
        self.root = root.resolve()
        # Map module names, such as plot_geometry, to source paths.
        self.module_files: dict[str, Path] = {}
        # Map fully qualified function/method names to definition records.
        self.functions: dict[str, FunctionRecord] = {}
        # Map class names to fully qualified class names when unique.
        self.class_defs: dict[str, str] = {}
        # Map class-qualified names to their method-qualified names.
        self.class_methods: dict[str, dict[str, str]] = defaultdict(dict)
        # Map class-qualified names to self.attribute -> class-qualified type.
        self.self_attr_types: dict[str, dict[str, str]] = defaultdict(dict)
        # Store __init__ nodes for type inference without reporting them as audit targets.
        self.init_records: list[FunctionRecord] = []
        # Map each function/method to repo-local functions it calls.
        self.call_graph: dict[str, set[str]] = defaultdict(set)
        # Store unresolved internal-looking calls for review.
        self.unresolved_by_function: dict[str, set[str]] = defaultdict(set)
        # Store notebook-cell entry points and unresolved calls.
        self.cells: list[CellRecord] = []

    def run(self):
        # Build the symbol table before attempting to resolve any calls.
        self._index_python_files()
        # Use indexed class names to resolve self.attribute assignments.
        self._index_self_attribute_types()
        # Resolve calls inside repository-defined functions/methods.
        self._build_call_graph()
        # Resolve direct repo-local calls in notebook cells.
        self._index_notebook_cells()
        # Traverse from notebook calls through repo-local subcalls.
        reachable = self._reachable_from_notebooks()
        # Print the cleanup-oriented report to stdout.
        self._print_report(reachable)

    def _iter_python_files(self):
        # os.walk lets us prune excluded folders before descending into them.
        for current_root, dirnames, filenames in os.walk(self.root):
            # Mutating dirnames in place prevents traversal into data/output/cache folders.
            dirnames[:] = sorted(name for name in dirnames if name not in EXCLUDED_DIRS)
            for filename in sorted(filenames):
                # Only Python source scripts define searchable functions/classes.
                if not filename.endswith(".py"):
                    continue
                # Exclude this reporting script from the analysis symbol index.
                if filename == SELF_FILE:
                    continue
                yield Path(current_root) / filename

    def _parse_file(self, path: Path):
        # Parse source without importing it, so no analysis code executes.
        try:
            # utf-8-sig handles files with a BOM without treating it as syntax.
            return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            print(f"[WARN] Could not parse {path.name}: {exc}")
            return None

    def _module_name(self, path: Path):
        # Current repository scripts are top-level modules named after the file stem.
        return path.stem

    def _index_python_files(self):
        # First pass records module files and all function/method definitions.
        for path in self._iter_python_files():
            module = self._module_name(path)
            self.module_files[module] = path
            tree = self._parse_file(path)
            if tree is None:
                continue
            import_aliases = self._collect_import_aliases(tree)
            self._index_tree_functions(path, module, tree, import_aliases)

    def _index_tree_functions(self, path: Path, module: str, tree: ast.AST, import_aliases: dict[str, str]):
        # Top-level functions are qualified as module.function.
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{module}.{node.name}"
                self.functions[qualified] = FunctionRecord(
                    qualified_name=qualified,
                    module=module,
                    file=path,
                    line=node.lineno,
                    node=node,
                    import_aliases=import_aliases,
                )
            elif isinstance(node, ast.ClassDef):
                # Class names are tracked so variable assignments can resolve instance methods.
                class_qualified = f"{module}.{node.name}"
                self.class_defs[node.name] = class_qualified
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # __init__ is used for type inference only, not as a cleanup target.
                        if method.name == "__init__":
                            self.init_records.append(FunctionRecord(
                                qualified_name=f"{module}.{node.name}.{method.name}",
                                module=module,
                                file=path,
                                line=method.lineno,
                                class_name=class_qualified,
                                node=method,
                                import_aliases=import_aliases,
                            ))
                            continue
                        qualified = f"{module}.{node.name}.{method.name}"
                        self.functions[qualified] = FunctionRecord(
                            qualified_name=qualified,
                            module=module,
                            file=path,
                            line=method.lineno,
                            class_name=class_qualified,
                            node=method,
                            import_aliases=import_aliases,
                        )
                        self.class_methods[class_qualified][method.name] = qualified

    def _collect_import_aliases(self, tree: ast.AST):
        # Import aliases let calls like pg.plot_x resolve to plot_geometry.plot_x.
        aliases: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for item in node.names:
                    module = item.name.split(".")[0]
                    aliases[item.asname or module] = module
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.split(".")[0]
                for item in node.names:
                    if item.name == "*":
                        continue
                    aliases[item.asname or item.name] = f"{module}.{item.name}"
        return aliases

    def _index_self_attribute_types(self):
        # Detect simple assignments such as self.calculator = SimpleCalculation().
        for record in self.init_records:
            # __init__ records are intentionally excluded from reachable/unused reports.
            if record.node is None or not record.class_name:
                continue
            for node in ast.walk(record.node):
                target = self._self_attr_assignment_target(node)
                if target is None:
                    continue
                assigned_class = self._assigned_class_name(node.value, record.import_aliases)
                if assigned_class is not None:
                    self.self_attr_types[record.class_name][target] = assigned_class

    def _self_attr_assignment_target(self, node: ast.AST):
        # Only simple self.attribute = ... assignments are useful for method-call resolution.
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            return None
        target = node.targets[0]
        if not isinstance(target, ast.Attribute):
            return None
        if not isinstance(target.value, ast.Name) or target.value.id != "self":
            return None
        return target.attr

    def _assigned_class_name(self, value: ast.AST, import_aliases: dict[str, str]):
        # Resolve the class used in Constructor(...) assignments when it is repo-local.
        if not isinstance(value, ast.Call):
            return None
        # Constructor resolution has no current-module preference in assignment contexts.
        resolved = self._resolve_name_or_attribute(value.func, import_aliases, {}, None, None)
        if resolved in self.class_defs.values():
            return resolved
        if isinstance(value.func, ast.Name) and value.func.id in self.class_defs:
            return self.class_defs[value.func.id]
        return None

    def _build_call_graph(self):
        # Resolve calls inside every indexed function/method.
        for qualified, record in self.functions.items():
            local_vars = self._collect_local_instance_assignments(record.node, record.import_aliases)
            # Plot modules receive PlotCreator as a self argument through wrapper calls.
            context_class = self._infer_context_class(record)
            for call in self._iter_calls(record.node):
                resolved = self._resolve_call(call.func, record, local_vars, context_class)
                if resolved in self.functions:
                    self.call_graph[qualified].add(resolved)
                elif self._looks_repo_like(call.func):
                    self.unresolved_by_function[qualified].add(self._call_text(call.func))

    def _collect_local_instance_assignments(self, node: ast.AST | None, import_aliases: dict[str, str]):
        # Track simple local assignments such as pc = PlotCreator().
        local_vars: dict[str, str] = {}
        if node is None:
            return local_vars
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign) or len(child.targets) != 1:
                continue
            if not isinstance(child.targets[0], ast.Name):
                continue
            assigned_class = self._assigned_class_name(child.value, import_aliases)
            if assigned_class is not None:
                local_vars[child.targets[0].id] = assigned_class
        return local_vars

    def _iter_calls(self, node: ast.AST | None):
        # Yield call nodes from an AST node, preserving static-only behavior.
        if node is None:
            return []
        return [child for child in ast.walk(node) if isinstance(child, ast.Call)]

    def _infer_context_class(self, record: FunctionRecord):
        # Class methods already know their owning class.
        if record.class_name is not None:
            return record.class_name
        # Free functions in plot_*.py are called with PlotCreator's self via wrappers.
        if record.module.startswith("plot_") and self._first_arg_is_self(record.node):
            return self.class_defs.get("PlotCreator")
        return None

    def _first_arg_is_self(self, node: ast.AST | None):
        # Detect module-level functions that intentionally receive a self-like object.
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        return bool(node.args.args and node.args.args[0].arg == "self")

    def _resolve_call(
            self,
            func: ast.AST,
            record: FunctionRecord,
            local_vars: dict[str, str],
            context_class: str | None,
    ):
        # Resolve one call expression in the context of a repository function.
        return self._resolve_name_or_attribute(
            func,
            record.import_aliases,
            local_vars,
            context_class,
            record.module,
        )

    def _resolve_name_or_attribute(
            self,
            func: ast.AST,
            import_aliases: dict[str, str],
            local_vars: dict[str, str],
            current_class: str | None,
            current_module: str | None,
    ):
        # Direct calls may be imported names or functions in the same module.
        if isinstance(func, ast.Name):
            return self._resolve_name(func.id, import_aliases, current_module)
        # Attribute calls cover module_alias.function, instance.method, and self.method.
        if isinstance(func, ast.Attribute):
            return self._resolve_attribute(func, import_aliases, local_vars, current_class)
        return None

    def _resolve_name(self, name: str, import_aliases: dict[str, str], current_module: str | None = None):
        # Imported repo-local functions/classes are resolved through the alias map.
        imported = import_aliases.get(name)
        if imported in self.functions or imported in self.class_defs.values():
            return imported
        # A same-module top-level function wins over ambiguous wrapper methods elsewhere.
        if current_module is not None:
            same_module = f"{current_module}.{name}"
            if same_module in self.functions:
                return same_module
        # Unique short function names can be resolved when there is no ambiguity.
        matches = [qualified for qualified in self.functions if qualified.endswith(f".{name}")]
        return matches[0] if len(matches) == 1 else None

    def _resolve_attribute(
            self,
            func: ast.Attribute,
            import_aliases: dict[str, str],
            local_vars: dict[str, str],
            current_class: str | None,
    ):
        # Resolve module_alias.function calls.
        if isinstance(func.value, ast.Name):
            base = func.value.id
            imported = import_aliases.get(base)
            if imported:
                candidate = f"{imported}.{func.attr}"
                if candidate in self.functions or candidate in self.class_defs.values():
                    return candidate
            # Resolve local_instance.method calls.
            if base in local_vars:
                return self.class_methods[local_vars[base]].get(func.attr)
            # Resolve self.method calls inside class methods.
            if base == "self" and current_class is not None:
                return self.class_methods[current_class].get(func.attr)
            # Resolve common repo object parameter names to their class methods.
            if base in KNOWN_VARIABLE_CLASSES:
                class_name = KNOWN_VARIABLE_CLASSES[base]
                repo_class = self.class_defs.get(class_name)
                if repo_class is not None:
                    return self.class_methods[repo_class].get(func.attr)
        # Resolve self.attribute.method calls using assignments discovered from __init__.
        if isinstance(func.value, ast.Attribute) and current_class is not None:
            base_attr = func.value
            if isinstance(base_attr.value, ast.Name) and base_attr.value.id == "self":
                assigned_class = self.self_attr_types[current_class].get(base_attr.attr)
                if assigned_class:
                    return self.class_methods[assigned_class].get(func.attr)
        return None

    def _looks_repo_like(self, func: ast.AST):
        # Suppress obvious external/library calls; unresolved names are only hints.
        text = self._call_text(func)
        if not text:
            return False
        # Ignore builtin/common calls that are not repository functions.
        if text in IGNORED_CALL_NAMES:
            return False
        # Ignore common list/dict-like methods even when the receiver is named self.*.
        if text.rsplit(".", 1)[-1] in IGNORED_METHOD_NAMES:
            return False
        external_prefixes = ("np.", "pd.", "plt.", "sns.", "scipy.", "stats.", "os.", "Path.")
        if text.startswith(external_prefixes):
            return False
        # Keep unresolved direct names only when they match a repo-defined function/method name.
        leaf_name = text.rsplit(".", 1)[-1]
        if any(qualified.endswith(f".{leaf_name}") for qualified in self.functions):
            return True
        # Keep self.* calls because they may indicate a missed class/member resolution.
        return text.startswith("self.")

    def _call_text(self, func: ast.AST):
        # Convert simple call expressions into readable text for unresolved reports.
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parent = self._call_text(func.value)
            return f"{parent}.{func.attr}" if parent else func.attr
        return ""

    def _index_notebook_cells(self):
        # Parse selected notebooks as JSON and analyze code cells only.
        for notebook in NOTEBOOKS:
            path = self.root / notebook
            if not path.exists():
                print(f"[WARN] Notebook not found: {notebook}")
                continue
            # Read notebook JSON only; code cells are not executed.
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            local_imports: dict[str, str] = {}
            local_vars: dict[str, str] = {}
            for cell_number, cell in enumerate(data.get("cells", []), start=1):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                cell_record = self._analyze_notebook_cell(
                    notebook,
                    cell_number,
                    source,
                    local_imports,
                    local_vars,
                )
                if cell_record.direct_calls or cell_record.unresolved:
                    self.cells.append(cell_record)

    def _analyze_notebook_cell(
            self,
            notebook: str,
            cell_number: int,
            source: str,
            local_imports: dict[str, str],
            local_vars: dict[str, str],
    ):
        # Invalid or magic-heavy notebook cells are skipped with a warning-like record.
        record = CellRecord(notebook=notebook, cell_number=cell_number)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            # Notebook magic/shell cells are not valid Python and cannot contain approved roots.
            if source.lstrip().startswith(("%", "!")):
                return record
            # Non-magic parse failures are kept only if they may hide a real root call.
            if any(root in source for root in PLOT_NOTEBOOK_ROOT_OBJECTS | EVALUATION_ROOT_FUNCTIONS):
                record.unresolved.add(f"Could not parse cell: {exc}")
            return record
        # Notebook imports and object assignments are cumulative across cells.
        local_imports.update(self._collect_import_aliases(tree))
        local_vars.update(self._collect_local_instance_assignments(tree, local_imports))
        # Resolve only approved root calls made in the cell body.
        for call in self._iter_calls(tree):
            if not self._is_allowed_notebook_root(notebook, call.func):
                continue
            # Notebook roots have no current-module context.
            resolved = self._resolve_name_or_attribute(call.func, local_imports, local_vars, None, None)
            if resolved in self.functions:
                record.direct_calls.add(resolved)
            elif self._looks_repo_like(call.func):
                record.unresolved.add(self._call_text(call.func))
        return record

    def _is_allowed_notebook_root(self, notebook: str, func: ast.AST):
        # WT and optogenetic notebooks only seed from PlotCreator and SurvivalStatsRunner objects.
        if notebook in {"WT-LP_and_ConGeo.ipynb", "OptogeneticFigure.ipynb"}:
            return (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in PLOT_NOTEBOOK_ROOT_OBJECTS
            )
        # Evaluation notebook only seeds from the three selected tracking_qc helpers.
        if notebook == "evaluation.ipynb":
            if isinstance(func, ast.Name):
                return func.id in EVALUATION_ROOT_FUNCTIONS
            if isinstance(func, ast.Attribute):
                return func.attr in EVALUATION_ROOT_FUNCTIONS
        # Any other notebook is outside the declared pipeline root set.
        return False

    def _reachable_from_notebooks(self):
        # Seed graph traversal with every direct repo-local call found in notebooks.
        reachable: set[str] = set()
        queue = deque(sorted({call for cell in self.cells for call in cell.direct_calls}))
        while queue:
            current = queue.popleft()
            if current in reachable:
                continue
            reachable.add(current)
            for child in sorted(self.call_graph.get(current, ())):
                if child not in reachable:
                    queue.append(child)
        return reachable

    def _closure_from_roots(self, roots: set[str]):
        # Calculate a per-cell transitive closure so notebook subcalls are visible.
        closure: set[str] = set()
        queue = deque(sorted(roots))
        while queue:
            current = queue.popleft()
            if current in closure:
                continue
            closure.add(current)
            for child in sorted(self.call_graph.get(current, ())):
                if child not in closure:
                    queue.append(child)
        return closure

    def _relative_location(self, qualified: str):
        # Format definition location as file:line for cleanup review.
        record = self.functions[qualified]
        rel_path = record.file.relative_to(self.root)
        return f"{rel_path}:{record.line}"

    def _print_report(self, reachable: set[str]):
        # Print notebook entry points first so the pipeline roots are visible.
        print("\nNotebook Direct Repo Calls")
        print("=" * 26)
        for cell in self.cells:
            print(f"\n{cell.notebook} cell {cell.cell_number}")
            if cell.direct_calls:
                for call in sorted(cell.direct_calls):
                    print(f"  USED {call}  ({self._relative_location(call)})")
                # Print repo-local descendants reached recursively from this cell's calls.
                descendants = self._closure_from_roots(cell.direct_calls) - cell.direct_calls
                if descendants:
                    print("  SUBCALLS")
                    for subcall in sorted(descendants):
                        print(f"    {subcall}  ({self._relative_location(subcall)})")
            if cell.unresolved:
                for call in sorted(cell.unresolved):
                    print(f"  UNRESOLVED {call}")

        # Print all functions that are reachable through recursive repo-local calls.
        print("\nReachable Repo Functions/Methods")
        print("=" * 32)
        for qualified in sorted(reachable):
            print(f"  {qualified}  ({self._relative_location(qualified)})")

        # Anything defined but not reachable is a cleanup candidate, not an automatic delete.
        unused = sorted(set(self.functions) - reachable)
        print("\nNot Reached From Current Notebook Pipeline")
        print("=" * 42)
        for qualified in unused:
            print(f"  {qualified}  ({self._relative_location(qualified)})")

        # Include unresolved repo-looking calls inside functions so static blind spots are explicit.
        unresolved_pairs = [
            (parent, child)
            for parent, children in self.unresolved_by_function.items()
            for child in children
            if parent in reachable
        ]
        if unresolved_pairs:
            print("\nUnresolved Calls Inside Reachable Functions")
            print("=" * 42)
            for parent, child in sorted(unresolved_pairs):
                print(f"  {parent} -> {child}")


def main():
    # Run from the current working directory so the script matches the active repo.
    analyzer = RepoCallAnalyzer(Path.cwd())
    analyzer.run()


if __name__ == "__main__":
    main()
