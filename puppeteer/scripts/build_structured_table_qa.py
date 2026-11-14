"""Build the versioned Table QA pilot from pinned upstream snapshots."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
from decimal import Decimal
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from routing_benchmark.v4.artifacts.manifest import validate_artifact_pack
from routing_benchmark.v4.artifacts.schema import sha256_file
from routing_benchmark.v4.domain_packets.builder import _Writer
from routing_benchmark.v4.execution.adapters.common import read_json, write_json
from routing_benchmark.v4.execution.adapters.table_qa import ADAPTER_ID, decimal_value, score_answer, typed_cell


DOMAIN = "structured_data_processing"
VERSION = "structured-table-qa-pilot.v1"
OUTPUT = PROJECT / "data/RoutingBench/structured_table_qa_v1_40"
CACHE = PROJECT / ".cache/routingbench_v4/table_qa"
WTQ = (1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 17, 19, 20, 21, 22, 24, 31, 32, 37, 40, 47)
HITAB = {
    "100": "c26dd7d31e73948ab9deb29ca1fc89ef",
    "1004": "2c52974788f4b3fa993e5876b239cbe9",
    "103_totto12249-2": "a34f6e729c2148fc1c2cd88acc04a460",
    "104_totto12294-0": "c1e6a83fb46cede5b8fdc81d863c7377",
    "1113": "0a2a15793b2790d2eb0098e755c44c47",
    "1114": "9466257bd11e7dc29371ced3999789eb",
    "1123": "d2e7519b453f2992c662dd41d1e273bc",
    "115": "9e723ea38e1aa2c604c4694a93e29ff5",
    "1153": "c2c16cedcf3741ef9fab9d533df88843",
    "115_totto13338-2": "266fb3a27de1acc27fede79072470b35",
    "1180": "7682b95b7c74164664a19a6e1eb1ba87",
    "1276": "13cf423a96e9cb9aff914b84e467ebaa",
    "1277": "ce6d779c9853cc40ba8ce39db2235c47",
    "1340": "269dd1fd590e4d39eadf91213551194e",
}
TABLEBENCH = {
    "2438086f1b3d0cb3f63d46f4c9eb8dbf": "elevation_sum",
    "8e11fccdc147a157e7d7e9471759e877": "filtered_mean",
    "3642133c0b09a25ffa48bd6356c3a58d": "filtered_sum",
    "9294abdf58d2fa73160b9131f16ec61d": "extreme_difference",
}
SOURCES = {
    "wtq": ("https://github.com/ppasupat/WikiTableQuestions", "CC-BY-SA-4.0", "wtq.zip"),
    "hitab": ("https://github.com/microsoft/HiTab", "Computational Use of Data Agreement v1.0", "hitab.zip"),
    "tablebench": ("https://huggingface.co/datasets/dadayechong/TableBench", "Apache-2.0", "tablebench.jsonl"),
}


def table_from_rows(headers, matrix, *, row_paths=None, title=None):
    width = len(headers)
    if not 1 <= width <= 30 or not 1 <= len(matrix) <= 150 or any(len(row) != width for row in matrix):
        raise ValueError("Unexpected table shape")
    columns = [{"id": f"c{i}", "header_path": path} for i, path in enumerate(headers, 1)]
    rows = [
        {"id": f"r{i + 1}", "values": {
            col["id"]: value for col, value in zip(columns, row)
        }, **({"header_path": row_paths[i]} if row_paths is not None else {})}
        for i, row in enumerate(matrix)
    ]
    return {
        "table_id": "table", "columns": columns, "rows": rows,
        "title": title, "hierarchical": any(len(path) > 1 for path in headers + (row_paths or [])),
    }


def labels(headers):
    return [[str(name).strip() or f"Column {i}"] for i, name in enumerate(headers, 1)]


def hierarchy_paths(root, width):
    paths = {}

    def walk(node, parents):
        name = str(node["name"]).strip()
        path = parents + [name] if name and name not in ("<TOP>", "<LEFT>") else parents
        index = node.get("line_idx")
        if index is not None:
            if not isinstance(index, int) or not 0 <= index < width or index in paths:
                raise ValueError(f"Invalid hierarchy index {index}")
            paths[index] = path
        for child in node.get("children_dict", []):
            walk(child, path)

    walk(root, [])
    if set(paths) != set(range(width)):
        raise ValueError("Incomplete header hierarchy")
    return [paths[i] for i in range(width)]


def wtq_records(path):
    prefix = "WikiTableQuestions-master/"
    split = "data/pristine-unseen-tables.tsv"
    with ZipFile(path) as archive:
        records = {row["id"]: row for row in csv.DictReader(
            StringIO(archive.read(prefix + split).decode("utf-8")), delimiter="\t"
        )}
        for number in WTQ:
            record = records[f"nu-{number}"]
            table_file = prefix + record["context"]
            header, *matrix = csv.reader(StringIO(archive.read(table_file).decode("utf-8")))
            question = record["utterance"].strip()
            if number == 5:
                question = re.sub(r"\bfist\b", "first", question)
            if number == 10:
                question += " Return the years as numbers in chronological order."
            if number == 9:
                question = "What is the earliest year with a first-place finish? Return the year as a number."
            if number == 21:
                question = "Which nation won the most gold medals? Exclude the Total row."
            yield (
                record["id"], question, table_from_rows(labels(header), matrix),
                [typed_cell(value) for value in record["targetValue"].split("|")],
                {"question": record["utterance"], "answer": record["targetValue"],
                 "table_member": table_file, "split": split},
            )


def hitab_records(path):
    split = "HiTab-main/data/test_samples.jsonl"
    with ZipFile(path) as archive, ZipFile(BytesIO(archive.read("HiTab-main/data/tables.zip"))) as tables:
        records = {row["id"]: row for row in map(json.loads, archive.read(split).splitlines())}
        for table_id, record_id in HITAB.items():
            record = records[record_id]
            if record["table_id"] != table_id:
                raise ValueError("Unexpected source table")
            member = f"tables/hmt/{table_id}.json"
            table = json.loads(tables.read(member))
            matrix = [[cell["value"] for cell in row] for row in table["data"]]
            columns = hierarchy_paths(table["top_root"], len(matrix[0]))
            rows = hierarchy_paths(table["left_root"], len(matrix))
            depth = max(map(len, rows))
            row_columns = [[f"Row label level {i}"] for i in range(1, depth + 1)]
            combined = [row + [None] * (depth - len(row)) + values for row, values in zip(rows, matrix)]
            question = record["question"].strip()
            question = {
                "104_totto12294-0": "Which club did Lodge play for in the Fourth Division in 1985-86?",
                "1113": "In the Canada total row, which age group of women has the highest surgical intervention rate?",
                "1114": "In the Canada total row, what percentage of colorectal cancer tumours received a surgical intervention?",
                "1123": "In the Canada total row, which age group has the lowest surgical intervention rate?",
                "115_totto13338-2": "List the three Anchorman films in the order of the film headers from left to right.",
                "1277": "Among the named types of care, which one has the highest percentage of fully met needs? Exclude the 'any' row.",
            }.get(table_id, question)
            answer = [typed_cell(value) for value in record["answer"]]
            if re.search(r"\b(percent|percentage|proportion)\b", question, re.I) and all(
                item["type"] == "number" for item in answer
            ):
                question += " Give the result as a percentage (unit %)."
                answer = [{**item, "unit": "%"} for item in answer]
            yield (
                table_id, question, table_from_rows(row_columns + columns, combined, row_paths=rows, title=table["title"]),
                answer,
                {"question": record["question"], "answer": record["answer"], "record_id": record_id,
                 "table_member": member, "split": split, "aggregation": record["aggregation"],
                 "linked_cells": record["linked_cells"], "answer_formulas": record["answer_formulas"]},
            )


def tablebench_records(path):
    records = {row["id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    for record_id, operation in TABLEBENCH.items():
        record = records[record_id]
        header, matrix = record["table"]["columns"], record["table"]["data"]
        number = decimal_value
        if operation == "elevation_sum":
            witness = sum(number(row[4]) for row in matrix)
        elif operation == "filtered_mean":
            values = [number(row[6]) for row in matrix if number(row[3]) >= Decimal("6.0")]
            witness = (sum(values) / len(values)).quantize(Decimal("0.01"))
        elif operation == "filtered_sum":
            witness = sum(number(row[4]) for row in matrix if number(row[2]) > 20000)
        else:
            values = [number(row[4]) for row in matrix]
            witness = max(values) - min(values)
        if witness != number(record["answer"]):
            raise ValueError(f"Source answer disagrees with independent calculation: {record_id}")
        question = record["question"]
        if operation == "filtered_mean":
            question += " Round to two decimal places."
        yield (
            record_id, question, table_from_rows(labels(header), matrix), [typed_cell(record["answer"])],
            {"question": record["question"], "answer": record["answer"], "split": "TQA_test",
             "independent_numeric_witness": str(witness), "qsubtype": record["qsubtype"]},
        )


def build(output, cache):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    loaders = {"wtq": wtq_records, "hitab": hitab_records, "tablebench": tablebench_records}
    expected_counts = {"wtq": 22, "hitab": 14, "tablebench": 4}
    index = []
    with TemporaryDirectory(dir=output.parent, prefix=".table-qa-") as directory:
        stage = Path(directory) / "packets"
        for source, (url, license_name, file) in SOURCES.items():
            path = cache / file
            source_lock = {"repository": url, "archive_sha256": sha256_file(path)}
            if path.suffix == ".zip":
                with ZipFile(path) as archive:
                    source_lock["revision"] = archive.comment.decode("ascii")
            for source_id, question, table, answer, provenance in loaders[source](path):
                packet_id = f"tableqa-{source}-{source_id.lower()}"
                gold = {"answer": answer, "order": "sequence" if len(answer) == 1 or
                        (source == "wtq" and source_id == "nu-10") or
                        (source == "hitab" and source_id == "115_totto13338-2") else "set"}
                if not score_answer(answer, gold) or score_answer([{"type": "text", "value": "invalid sentinel"}], gold):
                    raise ValueError(f"Evaluator sanity check failed for {packet_id}")
                writer = _Writer(stage / "packs" / DOMAIN / packet_id)
                for name, data, visibility in (
                    ("public/task.json", {"question": question, "answer_format": "typed_value_list_v1"}, "public"),
                    ("public/tables.json", {"tables": [table]}, "public"),
                    ("private/answer.json", gold, "private"),
                    ("private/source.json", {"source_id": source_id, **source_lock, **provenance}, "private"),
                ):
                    writer.add(name, json.dumps(data, ensure_ascii=False, indent=2) + "\n", visibility)
                manifest = writer.manifest(
                    packet_id=packet_id, domain=DOMAIN, task=question, adapter_id=ADAPTER_ID,
                    source_type=source, license_name=license_name,
                    execution_readiness="pending_runtime_verification",
                    metadata={"source_id": source_id, "hierarchical": table["hierarchical"]},
                )
                write_json(writer.pack_dir / "manifest.json", replace(manifest, generator_version=VERSION).to_dict())
                report = validate_artifact_pack(writer.pack_dir)
                if not report["valid"]:
                    raise ValueError(report)
                index.append({
                    "packet_id": packet_id, "source": source, "source_id": source_id,
                    "hierarchical": table["hierarchical"], "rows": len(table["rows"]),
                    "columns": len(table["columns"]), "manifest_sha256": sha256_file(writer.pack_dir / "manifest.json"),
                })
        counts = dict(Counter(row["source"] for row in index))
        if counts != expected_counts:
            raise ValueError(f"Wrong cohort counts: {counts}")
        write_json(stage / "packet_index.json", {"version": VERSION, "counts": counts, "packets": index})
        stage.rename(output)
    print(f"Built {len(index)} table QA packets at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    build(args.output, args.cache)
