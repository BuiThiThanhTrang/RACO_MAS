"""Run-linked audit events; IDs and serialization never consume policy RNG."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import subprocess

SCHEMA_VERSION = "1.0"
CALL_CONTEXT = ContextVar("audit_call_context", default=None)

def json_safe(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value

def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()

def redact(value):
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in
                    ("api_key", "authorization", "secret", "password")) and not k.endswith("_env")
                    else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)

def write_manifest(run_dir, config, **metadata):
    source = Path(__file__).resolve().parents[1]
    hashes = {}
    for folder in ("agent", "inference", "role_aware", "tasks", "model", "config", "prompts"):
        for path in sorted((source / folder).rglob("*")):
            if path.suffix in {".py", ".yaml", ".json", ".jsonl"} and path.is_file():
                hashes[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    git = {"commit": "unknown", "dirty": "unknown"}
    try:
        git["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source,
                                               stderr=subprocess.DEVNULL, text=True).strip()
        git["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source,
                                                   stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    result = dict(schema_version=SCHEMA_VERSION, runtime_version="path_sessions_v1",
                  config=redact(config), source_hashes=hashes, git=git,
                  provider_reproducibility="not_guaranteed", **metadata)
    result["manifest_hash"] = digest(result)
    write_json(Path(run_dir) / "audit" / "manifests" / (result["manifest_hash"] + ".json"), result)
    write_json(Path(run_dir) / "audit" / "manifest.json", result)
    return result

class AuditTrace:
    def __init__(self, directory, run_id, task_id, attempt_id, enabled=True):
        self.directory = Path(directory)
        self.context = dict(run_id=str(run_id), task_id=str(task_id), attempt_id=str(attempt_id))
        self.enabled = enabled
        self.events = []
        self.counters = {}
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def new_id(self, kind):
        self.counters[kind] = self.counters.get(kind, 0) + 1
        return f"{kind}-{self.counters[kind]:04d}"

    def emit(self, event_type, **fields):
        event = json_safe(dict(schema_version=SCHEMA_VERSION, **self.context,
                              event_seq=len(self.events) + 1, event_type=event_type, **fields))
        self.events.append(event)
        if self.enabled:
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def save(self, name, data):
        if self.enabled:
            write_json(self.directory / name, data)

    @contextmanager
    def call_scope(self, **fields):
        token = CALL_CONTEXT.set((self, fields))
        try:
            yield
        finally:
            CALL_CONTEXT.reset(token)

    def call_totals(self):
        calls = [e for e in self.events if e["event_type"] == "model_call_finished"]
        return dict(api_calls=len(calls), total_tokens=sum(e.get("tokens") or 0 for e in calls),
                    unknown_usage_calls=sum(e.get("tokens") is None for e in calls),
                    model_cost=sum(e.get("model_cost") or 0 for e in calls))

@contextmanager
def observe_provider_call(messages, model, model_size=0):
    """Instrument each actual provider attempt, including failed retries."""
    context = CALL_CONTEXT.get()
    receipt = {"tokens": None, "usage_source": "unavailable"}
    if not context:
        yield receipt
        return
    trace, fields = context
    call_id = trace.new_id("call")
    trace.emit("model_call_started", **fields, call_id=call_id, model=model,
               prompt_digest=digest(messages), messages=messages)
    try:
        yield receipt
    except BaseException as error:
        trace.emit("model_call_finished", **fields, call_id=call_id, model=model,
                   status="error", error_type=type(error).__name__, **receipt)
        raise
    else:
        tokens = receipt.get("tokens")
        trace.emit("model_call_finished", **fields, call_id=call_id, model=model,
                   status="ok", model_cost=None if tokens is None else 2 * model_size * tokens,
                   **receipt)
