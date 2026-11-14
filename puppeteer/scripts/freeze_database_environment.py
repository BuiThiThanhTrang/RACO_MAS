"""Create a credential-free, checksum-addressed database reference bundle."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'benchmarks/database_v1_20261031'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    files = set((ROOT / 'routing_benchmark').rglob('*.py'))
    files.update((ROOT / 'routing_benchmark').rglob('*.json'))
    files.update((ROOT / 'routing_benchmark').rglob('*.yaml'))
    files.update(p for p in (ROOT / 'data/RoutingBench/v4_domain_packets_rc1/packs/database_migration_integrity').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for relative in ['inference/__init__.py', 'inference/policy/decision_model_client.py',
                     'config/shared_environment/database_jev_16k.yaml',
                     'config/shared_environment/code_roles.json',
                     'requirements-shared-environment.txt', 'docs/database_benchmark_protocol_vi.md',
                     'scripts/run_shared_environment.py', 'scripts/evaluate_shared_environment.py',
                     'scripts/validate_shared_environment.py', 'scripts/export_shared_review.py']:
        files.add(ROOT / relative)
    entries = []
    archive = DEST / 'database_v1.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(files):
            data = path.read_bytes()
            name = 'project/' + path.relative_to(ROOT).as_posix()
            bundle.writestr(name, data)
            entries.append({'path': name, 'sha256': digest(data), 'bytes': len(data)})
    versions = {}
    for name in ['openai', 'httpx', 'jsonschema', 'PyYAML', 'python-dotenv']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {'freeze_id': DEST.name, 'scope': 'database_migration_integrity: existing 12 packets only',
              'archive': archive.name, 'archive_sha256': digest(archive.read_bytes()),
              'python': platform.python_version(), 'sqlite': sqlite3.sqlite_version,
              'installed_dependencies': versions, 'files': entries,
              'policy': 'Write once. Future changes require a new version; historical runs retain their original source hashes.'}
    (DEST / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with zipfile.ZipFile(archive) as bundle:
        assert all(digest(bundle.read(entry['path'])) == entry['sha256'] for entry in entries)
    print(json.dumps({'freeze_id': DEST.name, 'files': len(entries), 'archive_sha256': report['archive_sha256']}))


if __name__ == '__main__':
    main()
