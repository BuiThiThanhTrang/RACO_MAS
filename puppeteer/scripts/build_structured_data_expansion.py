"""Package original task text and workbook cases for the shared Structured Data pilot."""
from __future__ import annotations

import argparse
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
from zipfile import ZipFile

from openpyxl import load_workbook
from openpyxl.utils.cell import get_column_letter, range_boundaries
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from routing_benchmark.v4.artifacts.manifest import validate_artifact_pack
from routing_benchmark.v4.artifacts.schema import sha256_file
from routing_benchmark.v4.domain_packets.builder import _Writer
from routing_benchmark.v4.execution.adapters.common import write_json
from routing_benchmark.v4.execution.adapters.spreadsheetbench_python_oj import _values_equal

CACHE = PROJECT / ".cache/routingbench_v4"
CANONICAL = PROJECT / "data/RoutingBench/v4_beta_4d_82_20261102"
DOMAIN = "structured_data_processing"
OUTPUT = PROJECT / "data/RoutingBench/structured_data_easy_medium_40"
SOURCE_COUNTS = {"spreadsheetbench": (10, 10), "wtm": (6, 6), "sheetcopilot": (4, 4)}
SOURCE_LICENSES = {
    "spreadsheetbench": "CC-BY-SA-4.0",
    "wtm": "CDLA-Permissive-2.0",
    "sheetcopilot": "Upstream non-commercial dataset terms",
}
CELL_TASK = re.compile(
    r"\b(formula|fill|sum|total|average|count|extract|replace|deduplicate|duplicates?|"
    r"sort|lookup|calculate|copy|multiply|convert|filter|combine|split|match|remove)\w*\b", re.I
)
UNSUPPORTED = re.compile(
    r"\b(pivot|charts?|graphs?|macros? enabled|password|external workbook|power query|"
    r"conditional formatting|font|highlight|border|freeze|dropdown|drop.down)\b", re.I
)


def workbook_cells(raw):
    with ZipFile(BytesIO(raw)) as workbook_zip:
        if any(
            name.startswith(("xl/charts/", "xl/pivotTables/", "xl/externalLinks/"))
            for name in workbook_zip.namelist()
        ):
            raise ValueError("Structural or external workbook dependencies")
    workbook = load_workbook(BytesIO(raw), read_only=True, data_only=False)
    try:
        if not 1 <= len(workbook.sheetnames) <= 5:
            raise ValueError("Workbook sheet count outside pilot bounds")
        cells = {}
        for sheet in workbook:
            if sheet.max_row is None or sheet.max_column is None:
                raise ValueError("Missing workbook dimensions")
            if sheet.max_row > 600 or sheet.max_column > 60:
                raise ValueError("Workbook dimensions outside pilot bounds")
            for row, values in enumerate(sheet.values, 1):
                for column, value in enumerate(values, 1):
                    if value is not None and value != "":
                        if not isinstance(value, (str, float, int)) and hasattr(value, "text"):
                            raise ValueError("Unsupported array formula")
                        cells[sheet.title, row, column] = value
        return set(workbook.sheetnames), cells
    finally:
        workbook.close()


def scope_for(input_bytes, answer_bytes, target=None):
    input_sheets, initial = workbook_cells(input_bytes)
    answer_sheets, expected = workbook_cells(answer_bytes)
    if input_sheets != answer_sheets:
        raise ValueError("Sheet structure changed")
    changed = [
        (sheet, row, col) for sheet, row, col in initial.keys() | expected.keys()
        if not _values_equal(initial.get((sheet, row, col)), expected.get((sheet, row, col)))
    ]
    if target is None:
        if len({sheet for sheet, _, _ in changed}) != 1:
            raise ValueError("Expected one edited sheet")
        sheet = changed[0][0]
        rows = [row for _, row, _ in changed]
        columns = [col for _, _, col in changed]
        target = (sheet, min(columns), min(rows), max(columns), max(rows))
    sheet, min_col, min_row, max_col, max_row = target
    changed = [(s, r, c) for s, r, c in changed
               if s == sheet and min_col <= c <= max_col and min_row <= r <= max_row]
    if (
        sheet not in answer_sheets
        or (max_col - min_col + 1) * (max_row - min_row + 1) > 120
        or not changed
    ):
        raise ValueError("Changes outside one small target region")
    region = f"{get_column_letter(min_col)}{min_row}:{get_column_letter(max_col)}{max_row}"
    formulas = sum(
        isinstance(expected.get(key), str) and expected[key].startswith("=") for key in changed
    )
    return {"sheet": sheet, "range": region}, len(changed), formulas, len(input_sheets)


def candidate(source, source_id, task, cases, *, target=None, paths=()):
    if not 12 <= len(task) <= 700 or not CELL_TASK.search(task) or UNSUPPORTED.search(task):
        return None
    try:
        scopes = [scope_for(initial, answer, target) for initial, answer in cases]
    except (ValueError, KeyError, IndexError, TypeError, OSError):
        return None
    cells = max(item[1] for item in scopes)
    formulas = max(item[2] for item in scopes)
    sheets = max(item[3] for item in scopes)
    medium_signal = re.search(r"\b(lookup|match|duplicates?|group|filter|combine|if|except)\b", task, re.I)
    easy = cells <= 20 and formulas <= 12 and sheets <= 2 and not medium_signal
    return {
        "source": source, "source_id": source_id, "task": task, "cases": cases,
        "scopes": [{**item[0], "require_formula": bool(re.search(
            r"\b(formula|formulas)\b", task, re.I
        ))} for item in scopes], "changed_cells_max": cells,
        "formula_cells_max": formulas, "workbook_sheets_max": sheets,
        "difficulty": "easy" if easy else "medium", "source_members": list(paths),
    }


def spreadsheetbench(cache):
    folder = cache / "five_domains_20261101/KAKA22__SpreadsheetBench"
    used = {
        path.parent.name.removeprefix("sdp-spreadsheetbench-")
        for path in (CANONICAL / "packs" / DOMAIN).glob("*/manifest.json")
    }
    verified_path = folder / "spreadsheetbench_verified_400.tar.gz"
    full_path = folder / "spreadsheetbench_912_v0.1.tar.gz"
    with tarfile.open(verified_path) as archive:
        rows = json.load(archive.extractfile("spreadsheetbench_verified_400/dataset.json"))
    records = {}
    for row in rows:
        task = row["instruction"].strip()
        if str(row["id"]) in used or not row.get("answer_sheet") or not CELL_TASK.search(task) or UNSUPPORTED.search(task):
            continue
        position = row.get("answer_position", "").rsplit("!", 1)[-1].replace("$", "")
        try:
            c1, r1, c2, r2 = range_boundaries(position)
        except ValueError:
            continue
        if None not in (c1, r1, c2, r2) and (c2 - c1 + 1) * (r2 - r1 + 1) <= 120:
            records[str(row["id"])] = (task, (row["answer_sheet"], c1, r1, c2, r2))
    by_task = defaultdict(dict)
    with tarfile.open(full_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".xlsx"):
                continue
            path = Path(member.name)
            source_id = path.parent.name
            if source_id in records:
                match = re.fullmatch(rf"([123])_{re.escape(source_id)}_(input|answer)\.xlsx", path.name)
                if match:
                    by_task[source_id][int(match[1]), match[2]] = (member.name, archive.extractfile(member).read())
    for source_id, (task, target) in sorted(records.items()):
        files = by_task[source_id]
        if all((case, kind) in files for case in (1, 2, 3) for kind in ("input", "answer")):
            pairs = [(files[case, "input"], files[case, "answer"]) for case in (1, 2, 3)]
            yield candidate("spreadsheetbench", source_id, task, [(a[1], b[1]) for a, b in pairs],
                            target=target, paths=[n for pair in pairs for n, _ in pair])


def wtm(path):
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        prefix = "WTM-Bench-main/data/Backward_EnronFuse_v3/"
        metadata = json.loads(archive.read(prefix + "metadata/QuestionVariations_nonvisual.json"))["metadata"]
        for name in sorted(names):
            if not name.startswith(prefix + "workbooks/") or not name.endswith(".xlsx"):
                continue
            source_id = Path(name).stem
            answer = prefix + f"metadata/ExpectedWorkbook/{source_id}-File.xlsx"
            task = metadata.get(source_id, {}).get("value", {}).get("level_1", "")
            if task and answer in names:
                yield candidate("wtm", source_id, task, [(archive.read(name), archive.read(answer))],
                                paths=[name, answer])


def sheetcopilot(path):
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        prefix = "SheetCopilot-main/dataset/"
        workbook = load_workbook(BytesIO(archive.read(prefix + "dataset.xlsx")), read_only=True, data_only=True)
        try:
            for name, number, context, task, *_ in workbook.worksheets[0].iter_rows(min_row=2, values_only=True):
                if not isinstance(task, str) or not isinstance(number, int):
                    continue
                source_id = f"{number}_{name}"
                input_name = prefix + f"task_sheets/{name}.xlsx"
                gold_prefix = prefix + f"task_sheet_answers_v2/{name}/{source_id}/"
                golds = sorted(n for n in names if n.startswith(gold_prefix) and n.endswith(".xlsx"))
                if input_name not in names or len(golds) != 1:
                    continue
                check = golds[0].removesuffix(".xlsx") + "_check.yaml"
                if check not in names:
                    continue
                board = yaml.safe_load(archive.read(check))["check_board"]
                checks = {
                    (kind, prop)
                    for sheet in board.values()
                    for kind, flags in sheet.items()
                    for prop, enabled in flags.items()
                    if enabled is True
                }
                if checks != {("cells", "values")}:
                    continue
                instruction = " ".join(value.strip() for value in (context, task) if isinstance(value, str) and value.strip())
                yield candidate("sheetcopilot", source_id, instruction,
                                [(archive.read(input_name), archive.read(golds[0]))], paths=[input_name, golds[0], check])
        finally:
            workbook.close()


def build(output, cache):
    if output.exists():
        raise FileExistsError(output)
    expansion = cache / "structured_expansion"
    selected = []
    for source, entries in (
        ("spreadsheetbench", spreadsheetbench(cache)),
        ("wtm", wtm(expansion / "wtm.zip")),
        ("sheetcopilot", sheetcopilot(expansion / "sheetcopilot.zip")),
    ):
        pool = [item for item in entries if item]
        seen_inputs = set()
        print(source, {tier: sum(item["difficulty"] == tier for item in pool) for tier in ("easy", "medium")})
        for tier, count in zip(("easy", "medium"), SOURCE_COUNTS[source]):
            candidates = sorted(
                (item for item in pool if item["difficulty"] == tier),
                key=lambda item: (item["workbook_sheets_max"], item["changed_cells_max"], item["source_id"]),
            )
            chosen = []
            for item in candidates:
                initial = item["cases"][0][0]
                if initial in seen_inputs:
                    continue
                seen_inputs.add(initial)
                chosen.append(item)
                if len(chosen) == count:
                    break
            if len(chosen) < count:
                raise ValueError(f"{source}/{tier} has only {len(chosen)} distinct inputs; need {count}")
            selected.extend(chosen)
    template = CANONICAL / "packs" / DOMAIN / "sdp-spreadsheetbench-250-20/public/execution_contract.json"
    base_contract = json.loads(template.read_text(encoding="utf-8"))
    rows = []
    for item in selected:
        source, source_id, task = (item[key] for key in ("source", "source_id", "task"))
        packet_id = f"sdp-pilot-{source}-{source_id.lower()}"
        writer = _Writer(output / "packs" / DOMAIN / packet_id)
        case_ids = list(range(1, len(item["cases"]) + 1))
        contract = {
            **base_contract, "trusted_evaluator_case_ids": case_ids, "excluded_case_ids": [],
            "recalculate_before_grading": False,
            "scoring": {**base_contract["scoring"], "hard_pass": "All included cases pass",
                        "soft_score": "Passed cases / included cases"},
            "terminal_condition": "Fresh public validation passes",
        }
        for name, value, visibility in (
            ("public/task.json", {"instruction": task, "agent_pool_id": DOMAIN,
                                  "candidate": contract["candidate"], "public_case": {
                                      "case_id": 1, "input_workbook": "public/cases/case_1/input.xlsx"}},
             "public"),
            ("public/execution_contract.json", contract, "public"),
            ("public/validation_contract.json", {
                "kind": "workbook_target_cells",
                "public_validation_scope": "case_1_only",
                "comparison": "Checks target cells against reference values or formulas. "
                              "Formula text is compared directly; when formulas are not requested, "
                              "a numeric value matching the reference's cached result is also accepted.",
            }, "public"),
            ("private/evaluation_target.json", {"trusted_evaluator_case_ids": case_ids,
                                                 "cases": dict(zip(map(str, case_ids), item["scopes"]))}, "private"),
        ):
            writer.add(name, json.dumps(value, ensure_ascii=False, indent=2) + "\n", visibility)
        for case_id, (initial, answer) in enumerate(item["cases"], 1):
            writer.add(f"{'public' if case_id == 1 else 'private'}/cases/case_{case_id}/input.xlsx",
                       initial, "public" if case_id == 1 else "private")
            writer.add(f"private/cases/case_{case_id}/answer.xlsx", answer, "private")
        metadata = {k: v for k, v in item.items() if k not in ("task", "cases", "scopes")}
        manifest = writer.manifest(
            packet_id=packet_id, domain=DOMAIN, task=task, adapter_id=contract["adapter_id"],
            source_type=source, license_name=SOURCE_LICENSES[source],
            execution_readiness="pending_runtime_verification",
            metadata=metadata,
        )
        write_json(writer.pack_dir / "manifest.json", manifest.to_dict())
        if not validate_artifact_pack(writer.pack_dir)["valid"]:
            raise ValueError(f"Invalid artifact pack: {packet_id}")
        rows.append({"packet_id": packet_id, "task": task, "cases": case_ids, **metadata})
    roles = output / "agent_pools" / f"{DOMAIN}.json"
    roles.parent.mkdir(parents=True)
    shutil.copyfile(CANONICAL / "agent_pools" / roles.name, roles)
    archives = [expansion / "wtm.zip", expansion / "sheetcopilot.zip",
                *sorted((cache / "five_domains_20261101/KAKA22__SpreadsheetBench").glob("*.tar.gz"))]
    write_json(output / "selection.json", {"source_sha256": {p.name: sha256_file(p) for p in archives}, "packets": rows})
    print(f"Built {len(rows)} packets under {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    build(args.output, args.cache)
