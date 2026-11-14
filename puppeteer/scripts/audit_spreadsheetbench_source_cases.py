"""Inspect SpreadsheetBench source-native cases before packet migration.

The report is intentionally read-only.  It summarizes workbook structure and
input/answer differences without extracting source workbooks to the repository.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries


def json_value(value):
    if hasattr(value, "text"):
        return {"formula": value.text, "ref": getattr(value, "ref", None)}
    return value


def cell_style(cell) -> dict:
    fill = cell.fill.fgColor.rgb or cell.fill.fgColor.indexed or cell.fill.fgColor.theme
    return {
        "font": [cell.font.name, cell.font.sz, cell.font.bold, cell.font.italic],
        "fill": [cell.fill.fill_type, fill],
        "number_format": cell.number_format,
        "alignment": [cell.alignment.horizontal, cell.alignment.vertical, cell.alignment.wrap_text],
    }


def cell_state(cell) -> dict:
    return {"value": json_value(cell.value), "style": cell_style(cell)}


def workbook_diff(input_book, answer_book, limit: int) -> dict:
    changes = []
    counts = {"value_or_formula": 0, "style_only": 0, "added_sheet": 0, "removed_sheet": 0}
    all_sheets = list(dict.fromkeys(input_book.sheetnames + answer_book.sheetnames))
    for sheet_name in all_sheets:
        if sheet_name not in input_book.sheetnames:
            counts["added_sheet"] += 1
            continue
        if sheet_name not in answer_book.sheetnames:
            counts["removed_sheet"] += 1
            continue
        before_sheet = input_book[sheet_name]
        after_sheet = answer_book[sheet_name]
        max_row = max(before_sheet.max_row, after_sheet.max_row)
        max_column = max(before_sheet.max_column, after_sheet.max_column)
        for row in range(1, max_row + 1):
            for column in range(1, max_column + 1):
                before = cell_state(before_sheet.cell(row, column))
                after = cell_state(after_sheet.cell(row, column))
                if before == after:
                    continue
                kind = "value_or_formula" if before["value"] != after["value"] else "style_only"
                counts[kind] += 1
                if len(changes) < limit:
                    change = {
                        "sheet": sheet_name,
                        "cell": before_sheet.cell(row, column).coordinate,
                        "kind": kind,
                        "input_value": before["value"],
                        "answer_value": after["value"],
                    }
                    if before["style"] != after["style"]:
                        change["input_style"] = before["style"]
                        change["answer_style"] = after["style"]
                    changes.append(change)
    return {"counts": counts, "sample_changes": changes}


def range_values(workbook, selector: str):
    sheet_name, cell_range = selector.rsplit("!", 1)
    sheet_name = sheet_name.strip().strip("'")
    if sheet_name not in workbook.sheetnames:
        return None
    min_column, min_row, max_column, max_row = range_boundaries(cell_range)
    sheet = workbook[sheet_name]
    return [
        [json_value(sheet.cell(row, column).value) for column in range(min_column, max_column + 1)]
        for row in range(min_row, max_row + 1)
    ]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--source-id", action="append", required=True)
    parser.add_argument("--change-limit", type=int, default=80)
    parser.add_argument("--show-range", action="append", default=[])
    args = parser.parse_args()

    reports = []
    with tarfile.open(args.archive, "r:gz") as archive:
        records = json.loads(archive.extractfile("all_data_912_v0.1/dataset.json").read())
        by_id = {str(record["id"]): record for record in records}
        for source_id in args.source_id:
            cases = []
            for case_id in (1, 2, 3):
                prefix = f"all_data_912_v0.1/spreadsheet/{source_id}/{case_id}_{source_id}_"
                input_book = load_workbook(io.BytesIO(archive.extractfile(prefix + "input.xlsx").read()), data_only=False)
                answer_bytes = archive.extractfile(prefix + "answer.xlsx").read()
                answer_book = load_workbook(io.BytesIO(answer_bytes), data_only=False)
                answer_cached = load_workbook(io.BytesIO(answer_bytes), data_only=True)
                cases.append(
                    {
                        "case_id": case_id,
                        "input_sheets": [
                            {"name": sheet.title, "rows": sheet.max_row, "columns": sheet.max_column}
                            for sheet in input_book.worksheets
                        ],
                        "answer_sheets": [
                            {"name": sheet.title, "rows": sheet.max_row, "columns": sheet.max_column}
                            for sheet in answer_book.worksheets
                        ],
                        "diff": workbook_diff(input_book, answer_book, args.change_limit),
                        "ranges": {
                            selector: {
                                "input": range_values(input_book, selector),
                                "answer": range_values(answer_book, selector),
                                "answer_cached": range_values(answer_cached, selector),
                            }
                            for selector in args.show_range
                        },
                    }
                )
                input_book.close()
                answer_book.close()
                answer_cached.close()
            reports.append({"source_id": source_id, "metadata": by_id[source_id], "cases": cases})
    print(json.dumps(reports, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
