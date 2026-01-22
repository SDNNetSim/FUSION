"""Pydantic schemas for codebase explorer API."""

from pydantic import BaseModel


class ModuleNode(BaseModel):
    """A node in the module tree."""

    name: str
    path: str
    type: str  # 'package', 'module', 'file'
    description: str | None = None
    children: list["ModuleNode"] = []


class FunctionInfo(BaseModel):
    """Information about a function or method."""

    name: str
    line_number: int
    signature: str
    docstring: str | None = None
    is_method: bool = False


class ClassInfo(BaseModel):
    """Information about a class."""

    name: str
    line_number: int
    docstring: str | None = None
    methods: list[FunctionInfo] = []
    bases: list[str] = []


class FileContent(BaseModel):
    """Response containing file content and metadata."""

    path: str
    name: str
    content: str
    language: str
    line_count: int
    classes: list[ClassInfo] = []
    functions: list[FunctionInfo] = []
    imports: list[str] = []
    docstring: str | None = None


class ModuleTreeResponse(BaseModel):
    """Response containing the full module tree."""

    root: ModuleNode
    total_modules: int
    total_files: int
