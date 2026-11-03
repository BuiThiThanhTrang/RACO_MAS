from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping
import wave
import zipfile

from .schema import ArtifactFile, ArtifactManifest, canonical_hash, sha256_bytes
from .logic import (
    SCIENTIFIC_OPERATIONS,
    STRUCTURED_OPERATIONS,
    scientific_result,
    structured_transform,
)


PROJECT_AUTHORED_DOMAINS = (
    "transactional_service",
    "structured_data_processing",
    "cross_app_workflow",
    "code_debugging",
    "personal_device_tool",
    "scientific_procedure",
    "document_media_processing",
    "resource_constrained_workflow",
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _csv_bytes(rows: list[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _deterministic_zip(files: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return output.getvalue()


def _simple_pdf(text: str, *, page_count: int = 1) -> bytes:
    page_count = max(1, int(page_count))
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    page_texts = [f"Page {index + 1}\n{escaped}" for index in range(page_count)]
    page_object_numbers = list(range(3, 3 + page_count))
    content_object_numbers = list(range(3 + page_count, 3 + 2 * page_count))
    font_object_number = 3 + 2 * page_count
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{' '.join(f'{number} 0 R' for number in page_object_numbers)}] /Count {page_count} >>"
        ).encode("ascii"),
    ]
    for content_number in content_object_numbers:
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
                f"/Contents {content_number} 0 R >>"
            ).encode("ascii")
        )
    for page_text in page_texts:
        lines = page_text.splitlines() or [page_text]
        commands = ["BT", "/F1 12 Tf", "50 760 Td"]
        for index, line in enumerate(lines):
            if index:
                commands.append("0 -18 Td")
            commands.append(f"({line}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1", errors="replace")
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return bytes(output)


def _encoded_wav(text: str) -> bytes:
    sample_rate = 8000
    samples: list[int] = []
    for byte in text.encode("utf-8"):
        for shift in range(7, -1, -1):
            frequency = 1200 if (byte >> shift) & 1 else 700
            count = int(sample_rate * 0.025)
            samples.extend(
                int(9000 * math.sin(2 * math.pi * frequency * index / sample_rate))
                for index in range(count)
            )
        samples.extend([0] * int(sample_rate * 0.01))
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(struct.pack("<h", value) for value in samples))
    return output.getvalue()


def _slide_deck_bytes(titles: list[str]) -> bytes:
    from pptx import Presentation

    presentation = Presentation()
    for title in titles:
        slide = presentation.slides.add_slide(presentation.slide_layouts[5])
        slide.shapes.title.text = title
    raw = io.BytesIO()
    presentation.save(raw)
    normalized: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(raw.getvalue()), "r") as archive:
        for name in archive.namelist():
            normalized[name] = archive.read(name)
    return _deterministic_zip(normalized)


def _image_bytes(title: str, lines: list[str], *, chart: bool = False) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    draw.text((45, 35), title, fill="black")
    if chart:
        draw.line((100, 520, 900, 520), fill="black", width=3)
        draw.line((100, 520, 100, 120), fill="black", width=3)
        points = [(150, 450), (320, 370), (500, 300), (680, 225), (850, 160)]
        draw.line(points, fill="blue", width=5)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="red")
    y = 90
    for line in lines:
        draw.text((45, y), line, fill="black")
        y += 45
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _workflow_index(base_workflow_id: str) -> int:
    try:
        return int(base_workflow_id.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return int(canonical_hash(base_workflow_id, 6), 16) % 997 + 1


def _transactional(index: int, task: str) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, bytes]]:
    operations = (
        "change_subscription_tier", "cancel_order", "update_shipping_address",
        "correct_account_owner", "schedule_service_pause", "adjust_invoice_dispute",
        "apply_loyalty_credit", "change_billing_cycle", "restore_service",
        "merge_duplicate_records", "change_reservation", "create_warranty_replacement",
        "close_account",
    )
    operation = operations[(index - 1) % len(operations)]
    # The public task explicitly describes an eligible change. Fault variants are
    # introduced after execution and must not contradict this initial artifact.
    eligible = True
    record = {
        "account_id": f"ACC-{index:04d}",
        "status": "active" if eligible else "suspended",
        "record_type": operation,
        "current_value": f"current-{index}",
        "renewal_day": 15 + index % 10,
        "balance_due": 0 if eligible else 25,
    }
    request = {
        "request_id": f"REQ-{index:04d}",
        "operation": operation,
        "changes": {"requested_value": f"approved-{index}"},
        "effective_date": f"2026-11-{(index % 20) + 1:02d}",
    }
    policy = {
        "active_account_required": True,
        "maximum_balance_due": 0,
        "allowed_operations": [operation],
        "require_confirmation_fields": ["request_id", "effective_date", "operation"],
    }
    candidate = {
        "account_id": record["account_id"],
        "request_id": request["request_id"],
        "operation": operation,
        "changes": request["changes"],
        "effective_date": request["effective_date"],
        "eligible": eligible,
    }
    public = {
        "public/account_record.json": _json_bytes(record),
        "public/request.json": _json_bytes(request),
        "public/policy.json": _json_bytes(policy),
    }
    private = {
        "private/reference.json": _json_bytes(
            {"expected_candidate": candidate, "terminal_requires": "committed_confirmation", "task": task}
        ),
        "private/validator_config.json": _json_bytes(
            {"validator": "transactional.v1", "eligible": eligible}
        ),
    }
    fixture = {
        "fixtures/service_store.json": _json_bytes(
            {"account": record, "ledger": [], "confirmation": None}
        )
    }
    return public, private, fixture


def _structured(index: int, task: str) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, bytes]]:
    rows = [
        {
            "record_id": f"R{index}-{n}",
            "entity_id": f"E{(n + 1) // 2}",
            "version": 1 + n % 2,
            "group": "A" if n % 2 else "B",
            "quantity": n + index % 3,
            "unit_price": 7 + n,
            "discount": n % 2,
            "timestamp": f"2026-10-{n:02d}T0{n}:00:00Z",
            "category": (" Retail ", "WHOLESALE", "Unknown")[n % 3],
            "survey_score": None if n == 2 else n,
            "nested": {"note": f"note-{n}"},
        }
        for n in range(1, 5)
    ]
    operation = STRUCTURED_OPERATIONS[(index - 1) % len(STRUCTURED_OPERATIONS)]
    transformed = structured_transform(rows, operation)
    contract = {
        "operation": operation,
        "task": task,
        "output_format": "json",
    }
    return (
        {
            "public/source.json": _json_bytes(rows),
            "public/transformation_contract.json": _json_bytes(contract),
        },
        {
            "private/reference.json": _json_bytes({"expected_rows": transformed}),
            "private/validator_config.json": _json_bytes(
                {"validator": "structured_data.v1", "operation": operation, "expected_row_count": len(transformed)}
            ),
        },
        {"fixtures/workspace_state.json": _json_bytes({"published": False})},
    )


def _cross_app(index: int, task: str) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, bytes]]:
    source = {
        "source_id": f"SRC-{index:04d}",
        "full_name": f"Customer {index}",
        "email": f"customer{index}@example.test",
        "consent": True,
        "updated_at": f"2026-10-{(index % 20) + 1:02d}",
        "workflow_request": task,
    }
    contract = {
        "destination_fields": ["external_id", "display_name", "email", "consent", "source_updated_at"],
        "mapping": {
            "source_id": "external_id",
            "full_name": "display_name",
            "email": "email",
            "consent": "consent",
            "updated_at": "source_updated_at",
            "workflow_request": "workflow_request",
        },
        "idempotency_key": "external_id",
    }
    expected = {contract["mapping"][key]: value for key, value in source.items()}
    return (
        {
            "public/source_record.json": _json_bytes(source),
            "public/destination_contract.json": _json_bytes(contract),
        },
        {"private/reference.json": _json_bytes({"expected_destination": expected, "task": task})},
        {
            "fixtures/source_store.json": _json_bytes({"records": [source]}),
            "fixtures/destination_store.json": _json_bytes({"records": [], "audit_log": []}),
        },
    )


def _code(index: int, task: str) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, bytes]]:
    bug_types = (
        "parser_boundary", "off_by_one", "schema_validation", "stale_cache",
        "serialization_roundtrip", "retry_condition", "timezone_conversion",
        "null_default", "stable_sort", "permission_check", "aggregation_total",
        "api_payload", "state_transition",
    )
    bug_type = bug_types[(index - 1) % len(bug_types)]
    if bug_type in {"parser_boundary", "off_by_one"}:
        source = "def window(values, end):\n    return values[:end - 1]\n"
        fixed = "def window(values, end):\n    return values[:end]\n"
        case = {"call": "window([1, 2, 3], 2)", "expected": [1, 2]}
    elif bug_type in {"schema_validation", "null_default", "api_payload"}:
        source = "def display_name(value):\n    return value.strip()\n"
        fixed = "def display_name(value):\n    return '' if value is None else value.strip()\n"
        case = {"call": "display_name(None)", "expected": ""}
    elif bug_type == "stable_sort":
        source = "def rank(rows):\n    return sorted(rows, key=lambda row: (row['score'], row['id']))\n"
        fixed = "def rank(rows):\n    return sorted(rows, key=lambda row: row['score'])\n"
        case = {"call": "rank([{'id': 2, 'score': 1}, {'id': 1, 'score': 1}])", "expected_ids": [2, 1]}
    else:
        source = f"def resolve(value):\n    # BUG: {bug_type}\n    return value - 1\n"
        fixed = f"def resolve(value):\n    # FIXED: {bug_type}\n    return value\n"
        case = {"call": "resolve(3)", "expected": 3}
    test_script = (
        "import json\nfrom target import *\n"
        f"CASE = json.loads({json.dumps(json.dumps(case))})\n"
        "print(json.dumps({'case': CASE, 'loaded': True}, sort_keys=True))\n"
    )
    return (
        {
            "public/repo/target.py": source.encode("utf-8"),
            "public/repo/test_regression.py": test_script.encode("utf-8"),
            "public/patch_constraints.json": _json_bytes(
                {"bug_type": bug_type, "allowed_file": "target.py", "network": False, "task": task}
            ),
        },
        {
            "private/reference.py": fixed.encode("utf-8"),
            "private/reference.json": _json_bytes({"bug_type": bug_type, "test_case": case}),
        },
        {"fixtures/test_allowlist.json": _json_bytes({"commands": ["python test_regression.py"]})},
    )


def _device(index: int, task: str) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, bytes]]:
    settings = (
        "notification_schedule", "network_connection", "display_scale", "automation_rule",
        "accessibility_mode", "backup_target", "application_permission", "focus_schedule",
        "paired_device", "storage_cleanup", "privacy_control", "default_application",
        "battery_optimization",
    )
    setting = settings[(index - 1) % len(settings)]
    target: Any = True if setting != "font_scale" else 1.25
    state = {"device_id": f"DEV-{index:04d}", "settings": {setting: False if target is True else 1.0}, "revision": 1}
    request = {"setting": setting, "value": target, "postcondition": f"settings.{setting} == {json.dumps(target)}", "task": task}
    return (
        {"public/device_state.json": _json_bytes(state), "public/request.json": _json_bytes(request)},
        {"private/reference.json": _json_bytes({"expected_setting": setting, "expected_value": target})},
        {"fixtures/device_runtime.json": _json_bytes({"state": state, "operation_log": []})},
    )


def _scientific(index: int, task: str) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, bytes]]:
    baseline = 10 + index / 10
    rows = [{"replicate": n, "measurement": round(baseline + delta, 3), "unit": "mg/L"} for n, delta in enumerate((-0.2, 0.1, 0.0, 0.1), 1)]
    operation = SCIENTIFIC_OPERATIONS[(index - 1) % len(SCIENTIFIC_OPERATIONS)]
    protocol = {
        "operation": operation,
        "task": task,
        "unit": "mg/L",
        "control": baseline,
        "factor": 1.5,
        "threshold": baseline,
        "exclusion_max": baseline + 0.15,
    }
    expected = scientific_result(rows, protocol)
    return (
        {"public/measurements.csv": _csv_bytes(rows), "public/protocol.json": _json_bytes(protocol)},
        {"private/reference.json": _json_bytes({"expected_candidate": expected, "accepted": True})},
        {"fixtures/analysis_state.json": _json_bytes({"published": False})},
    )


def _document(index: int, task: str) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, bytes]]:
    subtype = ("invoice", "audio_instruction", "signed_form", "chart", "report", "media_transcript", "scanned_table", "annotated_image", "slide_deck", "mixed_bundle")[(index - 1) % 10]
    total_rule = None
    if subtype == "invoice":
        fields: dict[str, Any] = {"invoice_id": f"INV-{2026000 + index}", "vendor": f"Example Vendor {index}", "subtotal": 52.5, "tax": 5.25, "total": 57.75}
        media_path, media_type = "public/invoice.png", "image/png"
        media = _image_bytes("ROUTINGBENCH INVOICE", [f"{key}: {value}" for key, value in fields.items()])
        total_rule = "subtotal + tax"
    elif subtype in {"audio_instruction", "media_transcript"}:
        fields = {"instruction": "Add gloves, labels, and sample tubes to the checklist.", "checklist_items": ["gloves", "labels", "sample tubes"], "timestamp_seconds": 0}
        media_path, media_type = "public/instruction.wav", "audio/wav"
        media = _encoded_wav(fields["instruction"])
    elif subtype == "signed_form":
        fields = {"form_id": f"FORM-{index:03d}", "signer": f"Reviewer {index}", "signed": True, "section_count": 4}
        media_path, media_type = "public/signed_form.pdf", "application/pdf"
        media = _simple_pdf("SIGNED FORM\n" + "\n".join(f"{key}: {value}" for key, value in fields.items()))
    elif subtype == "chart":
        fields = {"series_name": "throughput", "points": [12, 18, 24, 31, 39], "x_axis": "trial", "y_axis": "items_per_minute"}
        media_path, media_type = "public/chart.png", "image/png"
        media = _image_bytes("THROUGHPUT BY TRIAL", ["series: throughput", "values: 12, 18, 24, 31, 39"], chart=True)
    elif subtype == "report":
        fields = {"page_count": 3, "sections": ["Background", "Method", "Result"], "ordering": [1, 2, 3]}
        media_path, media_type = "public/report.pdf", "application/pdf"
        media = _simple_pdf("MULTI-PAGE REPORT\nBackground\nMethod\nResult", page_count=3)
    elif subtype == "scanned_table":
        fields = {"rows": [{"item": "A", "amount": 10}, {"item": "B", "amount": 15}], "total": 25}
        media_path, media_type = "public/scanned_table.png", "image/png"
        media = _image_bytes("SCANNED TABLE", ["A | 10", "B | 15", "TOTAL | 25"])
    elif subtype == "annotated_image":
        fields = {"regions": [{"id": "R1", "label": "input", "bbox": [10, 20, 100, 80]}, {"id": "R2", "label": "output", "bbox": [140, 20, 250, 80]}]}
        media_path, media_type = "public/annotated_image.png", "image/png"
        media = _image_bytes("ANNOTATED REGIONS", ["R1: input [10,20,100,80]", "R2: output [140,20,250,80]"])
    elif subtype == "slide_deck":
        fields = {"slides": [{"title": "Overview"}, {"title": "Evidence"}, {"title": "Conclusion"}], "attachment_refs": ["ATT-01"]}
        media_path, media_type = "public/deck.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        media = _slide_deck_bytes([item["title"] for item in fields["slides"]])
    else:
        fields = {"selected_form": "request.txt", "fields": {"request_id": f"MIX-{index:03d}", "approved": True}}
        media_path, media_type = "public/document_bundle.zip", "application/zip"
        media = _deterministic_zip({"request.txt": _json_bytes(fields), "notes.txt": b"Supporting bundle notes.\n"})
    contract = {"required_fields": list(fields), "source": media_path, "media_type": media_type, "task": task, "artifact_subtype": subtype}
    if total_rule:
        contract["total_rule"] = total_rule
    return (
        {media_path: media, "public/extraction_contract.json": _json_bytes(contract)},
        {"private/reference.json": _json_bytes({"expected_fields": fields})},
        {"fixtures/media_index.json": _json_bytes({"source": media_path, "machine_readable_data": fields, "backend": "deterministic_extraction_fixture_v1"})},
    )


def _resource(index: int, task: str) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, bytes]]:
    budget = 5 + index % 3
    routes = [
        {"route_id": "direct", "steps": ["lookup", "verify"], "cost": 4, "meets_postcondition": True},
        {"route_id": "careful", "steps": ["inspect", "lookup", "verify"], "cost": 6, "meets_postcondition": True},
        {"route_id": "cheap_unverified", "steps": ["lookup"], "cost": 2, "meets_postcondition": False},
    ]
    feasible = [route for route in routes if route["cost"] <= budget and route["meets_postcondition"]]
    expected = min(feasible, key=lambda item: item["cost"]) if feasible else None
    return (
        {
            "public/tool_graph.json": _json_bytes({"routes": routes}),
            "public/task_contract.json": _json_bytes({"budget": budget, "required_postconditions": ["result_retrieved", "result_verified"], "task": task}),
        },
        {"private/reference.json": _json_bytes({"expected_route": expected, "feasible_route_ids": [r["route_id"] for r in feasible]})},
        {"fixtures/budget_ledger.json": _json_bytes({"initial": budget, "remaining": budget, "events": []})},
    )


BUILDERS = {
    "transactional_service": _transactional,
    "structured_data_processing": _structured,
    "cross_app_workflow": _cross_app,
    "code_debugging": _code,
    "personal_device_tool": _device,
    "scientific_procedure": _scientific,
    "document_media_processing": _document,
    "resource_constrained_workflow": _resource,
}


def _media_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".json": "application/json",
        ".csv": "text/csv",
        ".py": "text/x-python",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
        ".wav": "audio/wav",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".zip": "application/zip",
    }.get(suffix, "application/octet-stream")


def build_artifact_pack(
    *,
    output_root: str | Path,
    base_workflow_id: str,
    domain: str,
    task: str,
    seed: int = 42,
) -> ArtifactManifest:
    if domain not in BUILDERS:
        raise ValueError(f"Unsupported concrete-artifact domain: {domain}")
    pack_dir = Path(output_root) / domain / base_workflow_id
    pack_dir.mkdir(parents=True, exist_ok=True)
    public, private, fixtures = BUILDERS[domain](_workflow_index(base_workflow_id), task)
    groups = (("public", public), ("private", private), ("fixture", fixtures))
    records: dict[str, list[ArtifactFile]] = {"public": [], "private": [], "fixture": []}
    for visibility, files in groups:
        for relative_path, content in sorted(files.items()):
            destination = pack_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            artifact_id = relative_path.replace("/", "_").replace(".", "_")
            records[visibility].append(
                ArtifactFile(
                    artifact_id=artifact_id,
                    path=relative_path,
                    media_type=_media_type(relative_path),
                    sha256=sha256_bytes(content),
                    visibility=visibility,
                )
            )
    manifest = ArtifactManifest(
        base_workflow_id=base_workflow_id,
        domain=domain,
        task=task,
        adapter_id=f"{domain}.v1",
        public_files=tuple(records["public"]),
        private_files=tuple(records["private"]),
        fixture_files=tuple(records["fixture"]),
        seed=seed,
        metadata={"counterfactual_initial_artifact_shared": True},
    )
    (pack_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    expected_paths = {
        "manifest.json",
        *(item.path for item in manifest.public_files),
        *(item.path for item in manifest.private_files),
        *(item.path for item in manifest.fixture_files),
    }
    # Packs are builder-owned generated artifacts. Prune only stale files inside
    # this exact workflow directory so an older schema cannot leak into review.
    for existing in sorted(pack_dir.rglob("*"), reverse=True):
        if existing.is_file() and existing.relative_to(pack_dir).as_posix() not in expected_paths:
            existing.unlink()
        elif existing.is_dir() and not any(existing.iterdir()):
            existing.rmdir()
    return manifest
