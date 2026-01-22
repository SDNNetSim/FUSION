"""API routes for codebase exploration."""

import ast
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from fusion.api.schemas.codebase import (
    ClassInfo,
    FileContent,
    FunctionInfo,
    ModuleNode,
    ModuleTreeResponse,
)

router = APIRouter(prefix="/codebase", tags=["codebase"])

# Directories to exclude from scanning
EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    ".venv",
    "venv",
    "build",
    "dist",
    "*.egg-info",
}

# Files to exclude
EXCLUDED_FILES = {".DS_Store", ".gitignore", "*.pyc", "*.pyo"}


def _get_fusion_root() -> Path:
    """Get the root path of the FUSION package."""
    return Path(__file__).parent.parent.parent.parent


def _should_exclude(name: str) -> bool:
    """Check if a file or directory should be excluded."""
    if name in EXCLUDED_DIRS or name in EXCLUDED_FILES:
        return True
    if name.startswith("."):
        return True
    if name.endswith(".pyc") or name.endswith(".pyo"):
        return True
    return False


def _get_module_description(path: Path) -> str | None:
    """Extract module description from __init__.py or module docstring."""
    if path.is_dir():
        init_file = path / "__init__.py"
        if init_file.exists():
            try:
                content = init_file.read_text()
                tree = ast.parse(content)
                return ast.get_docstring(tree)
            except Exception:  # nosec B110
                pass  # Silently ignore unparseable files
    elif path.suffix == ".py":
        try:
            content = path.read_text()
            tree = ast.parse(content)
            return ast.get_docstring(tree)
        except Exception:  # nosec B110
            pass  # Silently ignore unparseable files
    return None


def _build_module_tree(path: Path, base_path: Path) -> ModuleNode | None:
    """Recursively build the module tree."""
    if _should_exclude(path.name):
        return None

    rel_path = str(path.relative_to(base_path))

    if path.is_dir():
        # Check if it's a Python package
        init_file = path / "__init__.py"
        is_package = init_file.exists()

        children = []
        for child in sorted(path.iterdir()):
            child_node = _build_module_tree(child, base_path)
            if child_node:
                children.append(child_node)

        # Only include directories that have Python content
        if not children and not is_package:
            return None

        return ModuleNode(
            name=path.name,
            path=rel_path,
            type="package" if is_package else "directory",
            description=_get_module_description(path),
            children=children,
        )
    elif path.suffix == ".py":
        return ModuleNode(
            name=path.name,
            path=rel_path,
            type="module",
            description=_get_module_description(path),
            children=[],
        )

    return None


def _count_modules(node: ModuleNode) -> tuple[int, int]:
    """Count total modules and files in the tree."""
    modules = 1 if node.type in ("package", "module") else 0
    files = 1 if node.type == "module" else 0

    for child in node.children:
        child_modules, child_files = _count_modules(child)
        modules += child_modules
        files += child_files

    return modules, files


def _parse_python_file(path: Path) -> tuple[list[ClassInfo], list[FunctionInfo], list[str], str | None]:
    """Parse a Python file and extract classes, functions, and imports."""
    content = path.read_text()
    tree = ast.parse(content)

    classes: list[ClassInfo] = []
    functions: list[FunctionInfo] = []
    imports: list[str] = []
    module_docstring = ast.get_docstring(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    methods.append(
                        FunctionInfo(
                            name=item.name,
                            line_number=item.lineno,
                            signature=_get_function_signature(item),
                            docstring=ast.get_docstring(item),
                            is_method=True,
                        )
                    )

            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(f"{_get_attribute_name(base)}")

            classes.append(
                ClassInfo(
                    name=node.name,
                    line_number=node.lineno,
                    docstring=ast.get_docstring(node),
                    methods=methods,
                    bases=bases,
                )
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            functions.append(
                FunctionInfo(
                    name=node.name,
                    line_number=node.lineno,
                    signature=_get_function_signature(node),
                    docstring=ast.get_docstring(node),
                    is_method=False,
                )
            )

    return classes, functions, imports, module_docstring


def _get_attribute_name(node: ast.Attribute) -> str:
    """Get the full name of an attribute access."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _get_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Generate a function signature string."""
    args = []

    # Regular args
    for i, arg in enumerate(node.args.args):
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"

        # Check for default value
        default_offset = len(node.args.args) - len(node.args.defaults)
        if i >= default_offset:
            default = node.args.defaults[i - default_offset]
            arg_str += f" = {ast.unparse(default)}"

        args.append(arg_str)

    # *args
    if node.args.vararg:
        vararg = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation:
            vararg += f": {ast.unparse(node.args.vararg.annotation)}"
        args.append(vararg)

    # **kwargs
    if node.args.kwarg:
        kwarg = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation:
            kwarg += f": {ast.unparse(node.args.kwarg.annotation)}"
        args.append(kwarg)

    signature = f"({', '.join(args)})"

    # Return type
    if node.returns:
        signature += f" -> {ast.unparse(node.returns)}"

    return signature


@router.get("/tree", response_model=ModuleTreeResponse)
def get_module_tree() -> ModuleTreeResponse:
    """
    Get the full module tree of the FUSION codebase.

    :returns: Module tree structure with metadata.
    """
    fusion_root = _get_fusion_root()
    fusion_pkg = fusion_root / "fusion"

    if not fusion_pkg.exists():
        raise HTTPException(status_code=404, detail="FUSION package not found")

    root = _build_module_tree(fusion_pkg, fusion_root)
    if not root:
        raise HTTPException(status_code=500, detail="Failed to build module tree")

    total_modules, total_files = _count_modules(root)

    return ModuleTreeResponse(
        root=root,
        total_modules=total_modules,
        total_files=total_files,
    )


@router.get("/file/{path:path}", response_model=FileContent)
def get_file_content(path: str) -> FileContent:
    """
    Get the content and metadata of a Python file.

    :param path: Relative path to the file within the FUSION package.
    :returns: File content with parsed metadata.
    """
    fusion_root = _get_fusion_root()
    file_path = fusion_root / path

    # Security: ensure path is within FUSION directory
    try:
        file_path = file_path.resolve()
        fusion_root.resolve()
        if not str(file_path).startswith(str(fusion_root.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path") from None

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    try:
        content = file_path.read_text()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}") from e

    # Determine language
    suffix = file_path.suffix.lower()
    language_map = {
        ".py": "python",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".ini": "ini",
        ".txt": "plaintext",
        ".sh": "shell",
        ".bash": "shell",
    }
    language = language_map.get(suffix, "plaintext")

    # Parse Python files for metadata
    classes: list[ClassInfo] = []
    functions: list[FunctionInfo] = []
    imports: list[str] = []
    docstring: str | None = None

    if suffix == ".py":
        try:
            classes, functions, imports, docstring = _parse_python_file(file_path)
        except SyntaxError:  # nosec B110
            pass  # Invalid Python syntax, just return content
        except Exception:  # nosec B110
            pass  # Other parsing errors

    return FileContent(
        path=path,
        name=file_path.name,
        content=content,
        language=language,
        line_count=content.count("\n") + 1,
        classes=classes,
        functions=functions,
        imports=imports,
        docstring=docstring,
    )


@router.get("/search")
def search_codebase(q: str, limit: int = 20) -> list[dict]:
    """
    Search for files, classes, or functions in the codebase.

    :param q: Search query.
    :param limit: Maximum results to return.
    :returns: List of matching items.
    """
    if not q or len(q) < 2:
        return []

    fusion_root = _get_fusion_root()
    fusion_pkg = fusion_root / "fusion"
    results: list[dict] = []
    query = q.lower()

    for root, dirs, files in os.walk(fusion_pkg):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not _should_exclude(d)]

        for file in files:
            if _should_exclude(file):
                continue

            file_path = Path(root) / file
            rel_path = str(file_path.relative_to(fusion_root))

            # Match file name
            if query in file.lower():
                results.append(
                    {
                        "type": "file",
                        "name": file,
                        "path": rel_path,
                        "match": "filename",
                    }
                )

            # For Python files, also search classes and functions
            if file.endswith(".py") and len(results) < limit:
                try:
                    content = file_path.read_text()
                    tree = ast.parse(content)

                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, ast.ClassDef):
                            if query in node.name.lower():
                                results.append(
                                    {
                                        "type": "class",
                                        "name": node.name,
                                        "path": rel_path,
                                        "line": node.lineno,
                                        "match": "class",
                                    }
                                )
                        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                            if query in node.name.lower():
                                results.append(
                                    {
                                        "type": "function",
                                        "name": node.name,
                                        "path": rel_path,
                                        "line": node.lineno,
                                        "match": "function",
                                    }
                                )
                except Exception:  # nosec B110
                    pass  # Skip files that can't be parsed

            if len(results) >= limit:
                break

        if len(results) >= limit:
            break

    return results[:limit]
