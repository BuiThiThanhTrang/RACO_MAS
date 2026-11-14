"""Freeze the data layer of the Structured Data workbook task cohort."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import re
import tarfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries

from build_structured_data_expansion import CACHE, CANONICAL, DOMAIN, OUTPUT, PROJECT, scope_for
from routing_benchmark.v4.artifacts.manifest import load_manifest, validate_artifact_pack
from routing_benchmark.v4.artifacts.schema import sha256_bytes, sha256_file
from routing_benchmark.v4.domain_packets.builder import _Writer
from routing_benchmark.v4.execution.adapters.common import read_json, write_json

PACKET_ROOT = PROJECT / "data/RoutingBench/structured_data_workbook_packets_v1"
VERSION = "structured-data-workbook-packet.v1"
DELIVERABLE = {
    "artifact_type": "workbook",
    "format": "xlsx",
    "description": "Submit the completed workbook after applying the requested changes to the supplied workbook.",
}
REPOSITORIES = {
    "spreadsheetbench": "https://huggingface.co/datasets/KAKA22/SpreadsheetBench",
    "wtm": "https://github.com/microsoft/WTM-Bench",
    "sheetcopilot": "https://github.com/BraveGroup/SheetCopilot",
}
ARCHIVE_DIR = CACHE / "five_domains_20261101/KAKA22__SpreadsheetBench"
VERIFIED = "spreadsheetbench_verified_400.tar.gz"
FULL = "spreadsheetbench_912_v0.1.tar.gz"
ARCHIVES = {
    VERIFIED: ARCHIVE_DIR / VERIFIED,
    FULL: ARCHIVE_DIR / FULL,
    "wtm.zip": CACHE / "structured_expansion/wtm.zip",
    "sheetcopilot.zip": CACHE / "structured_expansion/sheetcopilot.zip",
}
TASK_RECORDS = {
    "spreadsheetbench": (VERIFIED, "spreadsheetbench_verified_400/dataset.json"),
    "wtm": ("wtm.zip", "WTM-Bench-main/data/Backward_EnronFuse_v3/metadata/QuestionVariations_nonvisual.json"),
    "sheetcopilot": ("sheetcopilot.zip", "SheetCopilot-main/dataset/dataset.xlsx"),
}


def read_source_workbooks(wanted, verified_ids):
    """Collect required source bytes once per archive without extracting source code."""
    files = {}
    for name, path in ARCHIVES.items():
        if name.endswith(".zip"):
            with ZipFile(path) as archive:
                files[name] = {member: archive.read(member) for member in wanted[name]}
            continue
        with tarfile.open(path, "r|gz") as archive:
            files[name] = {
                member.name: archive.extractfile(member).read()
                for member in archive if member.isfile() and (
                    member.name in wanted[name] or (
                        name == VERIFIED
                        and Path(member.name).parent.name in verified_ids
                        and re.search(r"_(init|golden)\.xlsx$", member.name)
                    )
                )
            }
    return files


def original_packet(pack):
    manifest = load_manifest(pack)
    task = read_json(pack / "public/task.json")
    target = read_json(pack / "private/evaluation_target.json")
    provenance = read_json(pack / "fixtures/upstream_provenance.json")
    selection = read_json(pack / "fixtures/case_selection.json")
    cases = []
    for number in target["trusted_evaluator_case_ids"]:
        members = [item for item in provenance["members"] if item["case_id"] == number]
        by_kind = {item["kind"]: (
            VERIFIED if item["source_member"].startswith("spreadsheetbench_verified_400/") else FULL,
            item["source_member"],
            item["sha256"],
        ) for item in members}
        cases.append({"case_id": number, "source_files": by_kind})
    return {
        "pack": pack, "manifest": manifest, "source": "spreadsheetbench",
        "source_id": str(provenance["source_id"]), "instruction": task["instruction"],
        "cases": cases, "public_example": task.get("source_visible_example"),
        "oracle": {
            "source_target": target["answer_position"],
            "semantic_requirements": read_json(pack / "public/validation_contract.json"),
            "status": "documented_in_track_b",
        },
        "provenance_notes": [
            note for note in provenance["adaptations"]
            if not note.startswith("Normalize the model-facing deliverable")
        ],
        "excluded_cases": selection["excluded_cases"],
    }


def expansion_packet(pack, selected):
    manifest = load_manifest(pack)
    task = read_json(pack / "public/task.json")
    target = read_json(pack / "private/evaluation_target.json")
    source = selected["source"]
    members = selected["source_members"]
    archive = {"spreadsheetbench": FULL, "wtm": "wtm.zip", "sheetcopilot": "sheetcopilot.zip"}[source]
    cases = []
    for index, number in enumerate(target["trusted_evaluator_case_ids"]):
        scope = target["cases"][str(number)]
        cases.append({
            "case_id": number,
            "source_files": {
                kind: (archive, members[2 * index + offset], None)
                for offset, kind in enumerate(("input", "answer"))
            },
            "target_region": {"sheet": scope["sheet"], "range": scope["range"]},
        })
    return {
        "pack": pack, "manifest": manifest, "source": source, "source_id": selected["source_id"],
        "instruction": task["instruction"], "cases": cases, "excluded_cases": [],
        "oracle": {
            "status": "independent_reference_pending",
            "region_origin": ("upstream_answer_position" if source == "spreadsheetbench"
                              else "source_input_to_groundtruth_workbook_diff"),
            "source_checklist_member": members[2] if source == "sheetcopilot" else None,
        },
        "selection": {
            "difficulty_estimate": selected["difficulty"],
            **{key: selected[key] for key in ("changed_cells_max", "formula_cells_max", "workbook_sheets_max")},
        },
        "provenance_notes": [],
    }


def current_workbook(pack, number, kind):
    visibility = "public" if number == 1 and kind == "input" else "private"
    return pack / f"{visibility}/cases/case_{number}/{kind}.xlsx"


def package_packet(stage, record, source_files):
    previous = record["manifest"]
    source, source_id = record["source"], record["source_id"]
    writer = _Writer(stage / "packs" / DOMAIN / previous.base_workflow_id)
    source_archive, source_record = TASK_RECORDS[source]
    instruction = record["instruction"]
    public = {
        "instruction": instruction,
        "input_workbook": "public/cases/case_1/input.xlsx",
        "deliverable": DELIVERABLE,
    }
    if record.get("public_example"):
        public["source_visible_example"] = record["public_example"]
    writer.add("public/task.json", json.dumps(public, ensure_ascii=False, indent=2) + "\n", "public")
    is_expansion = "selection" in record
    case_specs = []
    for case in record["cases"]:
        number = case["case_id"]
        artifacts = {}
        for kind, (archive, member, expected_hash) in case["source_files"].items():
            data = source_files[archive][member]
            previous_data = current_workbook(record["pack"], number, kind).read_bytes()
            if data != previous_data or (expected_hash and sha256_bytes(data) != expected_hash):
                raise ValueError(f"Source byte mismatch for {previous.base_workflow_id} case {number} {kind}")
            if is_expansion and source == "spreadsheetbench" and number == 1:
                suffix = "_init.xlsx" if kind == "input" else "_golden.xlsx"
                archive = VERIFIED
                member, = [name for name in source_files[VERIFIED]
                           if Path(name).parent.name == source_id and name.endswith(suffix)]
                data = source_files[archive][member]
            visibility = "public" if number == 1 and kind == "input" else "private"
            relative = f"{visibility}/cases/case_{number}/{kind}.xlsx"
            writer.add(relative, data, visibility)
            artifacts[kind] = {
                "path": relative, "archive": archive, "member": member,
                "sha256": sha256_bytes(data), "visibility": visibility,
            }
        conflicting_gold = source == "spreadsheetbench" and source_id == "13-1" and number in (2, 3)
        status = ("held_upstream_answer_conflict" if conflicting_gold else
                  "independent_reference_pending" if is_expansion else "prior_semantic_reference_verified")
        case_specs.append({
            "case_id": number,
            "role": "public_development" if number == 1 else "private_evaluation",
            "groundtruth_status": status,
            **({"reason": "Full-archive answers preserve first occurrence; the Verified task requires sorting by date and reference."}
               if conflicting_gold else {}),
            **({"target_region": case["target_region"]} if "target_region" in case else {}),
            **artifacts,
        })
    notes = record["provenance_notes"]
    if is_expansion and source == "spreadsheetbench":
        notes = ["Use the Verified-400 input and answer bytes for case 1 and full-archive cases 2–3."]
    spec = {
        "packet_version": VERSION,
        "packet_id": previous.base_workflow_id,
        "source": {
            "dataset": source, "repository": REPOSITORIES[source], "record_id": source_id,
            "license": previous.license, "instruction_archive": source_archive, "instruction_member": source_record,
            "source_packet_manifest_sha256": sha256_file(record["pack"] / "manifest.json"),
            "notes": notes,
        },
        "output": DELIVERABLE,
        "oracle": record["oracle"],
        "cases": case_specs,
        "excluded_source_cases": record["excluded_cases"],
        "selection": record.get("selection", {"stratum": "existing_track_b"}),
    }
    writer.add("private/packet_spec.json", json.dumps(spec, ensure_ascii=False, indent=2) + "\n", "private")
    manifest = writer.manifest(
        packet_id=previous.base_workflow_id, domain=DOMAIN, task=instruction,
        adapter_id="structured_data.packet_only.v1", source_type=source, license_name=previous.license,
        execution_readiness="pending_runtime_verification",
        metadata={"packet_version": VERSION, "source_id": source_id, "stage": "packet_constructed"},
    )
    write_json(writer.pack_dir / "manifest.json", replace(manifest, generator_version=VERSION).to_dict())
    return {
        "packet_id": previous.base_workflow_id, "source": source, "source_id": source_id,
        "selection": spec["selection"], "case_count": len(case_specs),
        "case_statuses": {str(case["case_id"]): case["groundtruth_status"] for case in case_specs},
        "manifest_sha256": sha256_file(writer.pack_dir / "manifest.json"),
    }


def package(output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    selection = read_json(OUTPUT / "selection.json")
    records = [original_packet(path.parent) for path in sorted((CANONICAL / "packs" / DOMAIN).glob("*/manifest.json"))]
    if len(records) != 20 or len(selection["packets"]) != 40:
        raise ValueError("Expected 20 Track B packets and 40 expansion packets")
    records.extend(expansion_packet(OUTPUT / "packs" / DOMAIN / row["packet_id"], row) for row in selection["packets"])
    if len({(record["source"], record["source_id"]) for record in records}) != len(records):
        raise ValueError("Duplicate source task")
    wanted = {name: set() for name in ARCHIVES}
    for record in records:
        for case in record["cases"]:
            for archive, member, _ in case["source_files"].values():
                wanted[archive].add(member)
    verified_ids = {record["source_id"] for record in records if "selection" in record and record["source"] == "spreadsheetbench"}
    archive_hashes = {name: sha256_file(path) for name, path in ARCHIVES.items()}
    if archive_hashes != selection["source_sha256"]:
        raise ValueError("The source archives differ from those pinned by the expansion")
    source_files = read_source_workbooks(wanted, verified_ids)
    sources = {name: {"sha256": digest} for name, digest in archive_hashes.items()}
    for name in ("wtm.zip", "sheetcopilot.zip"):
        with ZipFile(ARCHIVES[name]) as archive:
            revision = archive.comment.decode("ascii")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Expected a pinned Git revision in {name}")
        sources[name]["revision"] = revision
    for name in (VERIFIED, FULL):
        sources[name]["revision"] = "ab0b742b0fc95b946f212d80ac7771b5531272e4"
    with TemporaryDirectory(dir=output.parent, prefix=".structured-data-packet-stage-") as directory:
        stage = Path(directory) / "packets"
        rows = [package_packet(stage, record, source_files) for record in records]
        write_json(stage / "packet_index.json", {
            "packet_version": VERSION,
            "stage": "packet_constructed",
            "sources": sources,
            "packet_counts": dict(Counter(row["source"] for row in rows)),
            "case_count": sum(row["case_count"] for row in rows),
            "case_status_counts": dict(Counter(status for row in rows for status in row["case_statuses"].values())),
            "packets": rows,
        })
        verify(stage)
        stage.rename(output)
    print(f"Packaged {len(rows)} workbook tasks and {sum(row['case_count'] for row in rows)} source cases at {output}")


def upstream_instructions():
    with tarfile.open(ARCHIVES[VERIFIED]) as archive:
        rows = json.load(archive.extractfile(TASK_RECORDS["spreadsheetbench"][1]))
    with ZipFile(ARCHIVES["wtm.zip"]) as archive:
        wtm = json.loads(archive.read(TASK_RECORDS["wtm"][1]))["metadata"]
    with ZipFile(ARCHIVES["sheetcopilot.zip"]) as archive:
        sheetcopilot = archive.read(TASK_RECORDS["sheetcopilot"][1])
    tasks = {("spreadsheetbench", str(row["id"])): row["instruction"].strip() for row in rows}
    tasks.update({
        ("wtm", source_id): row["value"]["level_1"].strip()
        for source_id, row in wtm.items() if row.get("value", {}).get("level_1")
    })
    book = load_workbook(BytesIO(sheetcopilot), read_only=True, data_only=True)
    try:
        for name, number, context, task, *_ in book.worksheets[0].iter_rows(min_row=2, values_only=True):
            if isinstance(task, str) and isinstance(number, int):
                tasks["sheetcopilot", f"{number}_{name}"] = " ".join(
                    value.strip() for value in (context, task) if isinstance(value, str) and value.strip()
                )
    finally:
        book.close()
    return tasks


def verify(root):
    index = read_json(root / "packet_index.json")
    tasks = upstream_instructions()
    statuses = Counter()
    actual = {p.parent.name for p in (root / "packs" / DOMAIN).glob("*/manifest.json")}
    expected = {row["packet_id"] for row in index["packets"]}
    if actual != expected or len(actual) != 60:
        raise ValueError("Packet list does not match the 60-task index")
    for row in index["packets"]:
        pack = root / "packs" / DOMAIN / row["packet_id"]
        manifest = load_manifest(pack)
        public = read_json(pack / "public/task.json")
        spec = read_json(pack / "private/packet_spec.json")
        visibility = {item.path: item.visibility for item in (*manifest.public_files, *manifest.private_files)}
        if {p for p, mode in visibility.items() if mode == "public"} != {"public/task.json", public["input_workbook"]}:
            raise ValueError(f"Unexpected public files in {pack}")
        if public["instruction"].strip() != tasks[row["source"], row["source_id"]]:
            raise ValueError(f"Task text differs from upstream in {pack}")
        if public["deliverable"] != DELIVERABLE or spec["output"] != DELIVERABLE:
            raise ValueError(f"Unexpected deliverable in {pack}")
        if sha256_file(pack / "manifest.json") != row["manifest_sha256"] or not validate_artifact_pack(pack)["valid"]:
            raise ValueError(f"Manifest or asset integrity failure in {pack}")
        for case in spec["cases"]:
            statuses[case["groundtruth_status"]] += 1
            for kind in ("input", "answer"):
                artifact = case[kind]
                expected_visibility = "public" if kind == "input" and case["case_id"] == 1 else "private"
                if visibility.get(artifact["path"]) != expected_visibility or artifact["visibility"] != expected_visibility:
                    raise ValueError(f"Case visibility mismatch in {pack}")
                if sha256_file(pack / artifact["path"]) != artifact["sha256"]:
                    raise ValueError(f"Case checksum mismatch in {pack}")
            if "target_region" in case:
                region = case["target_region"]
                target = (region["sheet"], *range_boundaries(region["range"]))
                scope_for((pack / case["input"]["path"]).read_bytes(),
                          (pack / case["answer"]["path"]).read_bytes(), target)
    if dict(statuses) != index["case_status_counts"] or sum(statuses.values()) != index["case_count"]:
        raise ValueError("Case counts differ from the index")
    print(f"Verified {len(actual)} source instructions and manifests, {sum(statuses.values())} case pairs, and public/private scope.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PACKET_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output)
    else:
        package(args.output)
