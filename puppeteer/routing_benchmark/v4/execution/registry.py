from __future__ import annotations

from routing_benchmark.v4.artifacts.schema import ArtifactManifest
from .adapters.database import DatabaseMigrationAdapter
from .adapters.formal_proof import FormalProofAdapter

from .adapters import (
    CodeDebuggingAdapter,
    ExternalCodeDebuggingAdapter,
    CrossAppWorkflowAdapter,
    DocumentMediaProcessingAdapter,
    PersonalDeviceToolAdapter,
    ResourceConstrainedWorkflowAdapter,
    ScientificProcedureAdapter,
    StructuredDataProcessingAdapter,
    SpreadsheetBenchPythonOJAdapter,
    TransactionalServiceAdapter,
)


ADAPTERS = {
    "database_migration.sqlite.v1": DatabaseMigrationAdapter,
    "formal_proof.lean4.v1": FormalProofAdapter,
    "transactional_service.v1": TransactionalServiceAdapter,
    "structured_data_processing.v1": StructuredDataProcessingAdapter,
    "spreadsheetbench.python_oj.v1": SpreadsheetBenchPythonOJAdapter,
    "cross_app_workflow.v1": CrossAppWorkflowAdapter,
    "code_debugging.v1": CodeDebuggingAdapter,
    "code_debugging.external.v1": ExternalCodeDebuggingAdapter,
    "personal_device_tool.v1": PersonalDeviceToolAdapter,
    "scientific_procedure.v1": ScientificProcedureAdapter,
    "document_media_processing.v1": DocumentMediaProcessingAdapter,
    "resource_constrained_workflow.v1": ResourceConstrainedWorkflowAdapter,
}

# Only adapters backed by their authoritative domain runtime may be used by the
# live benchmark entry point. Synthetic v4 adapters remain registered for
# historical tracks, but are not silently promoted to live evaluation.
LIVE_ADAPTER_IDS = frozenset(
    {
        "database_migration.sqlite.v1",
        "formal_proof.lean4.v1",
        "spreadsheetbench.python_oj.v1",
        "code_debugging.external.v1",
    }
)


def adapter_for_manifest(manifest: ArtifactManifest):
    try:
        return ADAPTERS[manifest.adapter_id]()
    except KeyError as error:
        raise ValueError(f"No execution adapter registered for {manifest.adapter_id}") from error


def live_adapter_for_manifest(manifest: ArtifactManifest):
    if manifest.adapter_id not in LIVE_ADAPTER_IDS:
        raise ValueError(
            f"Adapter {manifest.adapter_id} is registered but is not enabled for live trajectories"
        )
    return adapter_for_manifest(manifest)
