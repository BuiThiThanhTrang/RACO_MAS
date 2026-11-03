"""Restricted runner for model-authored SpreadsheetBench transformations.

The worker intentionally exposes only a small Python/OpenPyXL surface.  It is
not a general-purpose Python sandbox: source is AST-checked, imports and dunder
access are removed/rejected, workbook loading/saving is path-guarded, and the
parent process supplies the wall-clock timeout and output-size limit.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from copy import copy
import json
from pathlib import Path
import re
from types import SimpleNamespace

from openpyxl import load_workbook as _openpyxl_load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.workbook.workbook import Workbook


_BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint", "help",
}
class CandidatePolicyError(ValueError):
    pass


class _ImportNormalizer(ast.NodeTransformer):
    def visit_Import(self, node: ast.Import):
        if any(alias.name not in {"openpyxl", "re"} for alias in node.names):
            raise CandidatePolicyError("Only import openpyxl and import re are accepted")
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom):
        names = {alias.name for alias in node.names}
        allowed = {
            "openpyxl": {"load_workbook"},
            "openpyxl.styles": {"Alignment", "Border", "Font", "PatternFill", "Side"},
            "collections": {"defaultdict"},
            "copy": {"copy"},
        }
        if node.module not in allowed or not names or not names.issubset(allowed[node.module]):
            raise CandidatePolicyError("Only documented OpenPyXL style helpers, collections.defaultdict and copy.copy are accepted")
        return None


def normalize_and_validate_source(source: str) -> ast.Module:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as error:
        raise CandidatePolicyError(f"SyntaxError: {error.msg} at line {error.lineno}") from error
    tree = _ImportNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)):
            raise CandidatePolicyError(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and (node.id in _BANNED_NAMES or "__" in node.id):
            raise CandidatePolicyError(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or "__" in node.attr):
            raise CandidatePolicyError(f"Forbidden attribute: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _BANNED_NAMES:
            raise CandidatePolicyError(f"Forbidden call: {node.func.id}")
    solve = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"]
    if len(solve) != 1:
        raise CandidatePolicyError("Candidate must define exactly one solve(input_path, output_path) function")
    if len(solve[0].args.args) != 2:
        raise CandidatePolicyError("solve must accept exactly input_path and output_path")
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.Expr)):
            raise CandidatePolicyError("Top-level execution is not allowed")
        if isinstance(node, ast.Expr) and not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            raise CandidatePolicyError("Only a module docstring is allowed at top level")
    return tree


def run_candidate(source_path: Path, input_path: Path, output_path: Path) -> dict:
    source_path = source_path.resolve()
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = normalize_and_validate_source(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loaded: list[Workbook] = []
    original_save = Workbook.save

    def guarded_load_workbook(filename, *args, **kwargs):
        if Path(filename).resolve() != input_path:
            raise CandidatePolicyError("Candidate may load only the supplied input_path")
        workbook = _openpyxl_load_workbook(input_path, *args, **kwargs)
        loaded.append(workbook)
        return workbook

    def guarded_save(workbook, filename):
        if workbook not in loaded:
            raise CandidatePolicyError("Candidate may save only a workbook loaded from input_path")
        if Path(filename).resolve() != output_path:
            raise CandidatePolicyError("Candidate may save only to the supplied output_path")
        return original_save(workbook, output_path)

    safe_builtins = {
        "True": True, "False": False, "None": None,
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "chr": chr, "divmod": divmod, "enumerate": enumerate, "float": float, "int": int, "len": len,
        "isinstance": isinstance, "iter": iter, "map": map, "next": next,
        "list": list, "max": max, "min": min, "range": range,
        "reversed": reversed, "round": round, "set": set, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "zip": zip,
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    }
    namespace = {
        "__builtins__": safe_builtins,
        "load_workbook": guarded_load_workbook,
        "openpyxl": SimpleNamespace(
            load_workbook=guarded_load_workbook,
            styles=SimpleNamespace(Alignment=Alignment, Border=Border, Font=Font, PatternFill=PatternFill, Side=Side),
        ),
        "Alignment": Alignment,
        "Border": Border,
        "Font": Font,
        "PatternFill": PatternFill,
        "Side": Side,
        "defaultdict": defaultdict,
        "copy": copy,
        "re": re,
    }
    Workbook.save = guarded_save
    try:
        exec(compile(tree, str(source_path), "exec"), namespace, namespace)
        namespace["solve"](str(input_path), str(output_path))
    finally:
        Workbook.save = original_save
    if not output_path.is_file():
        raise CandidatePolicyError("Candidate did not create output_path")
    workbook = _openpyxl_load_workbook(output_path, read_only=True, data_only=False)
    sheets = list(workbook.sheetnames)
    workbook.close()
    return {"output_path": str(output_path), "sheets": sheets, "output_bytes": output_path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = {"status": "success", **run_candidate(args.candidate, args.input, args.output)}
        print(json.dumps(report, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
