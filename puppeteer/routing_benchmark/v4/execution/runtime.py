"""Resolve domain toolchains without changing routing policy."""
import json
import os
from pathlib import Path
import re


def resolve_runtime(config, manifest_path, project):
    def resolve(value):
        path=Path(value)
        return path.resolve() if path.is_absolute() else (project/path).resolve()
    runtime=dict(config.get('runtime',{}))
    if config.get('runtime_profiles'):
        profiles=json.loads(resolve(config['runtime_profiles']).read_text(encoding='utf-8'))
        runtime.update(profiles.get(Path(manifest_path).parent.name,{}))
    for key in ('project_dir','lean_executable','repository_mirror','python_executable'):
        if key in runtime: runtime[key]=str(resolve(runtime[key]))
    if config.get('domain')=='formal_proof' and config.get('lean_cache'):
        cache=resolve(config['lean_cache'])
        contract=json.loads((Path(manifest_path).parent/'public/execution_contract.json').read_text(encoding='utf-8'))
        version=re.search(r'Lean\s+(\d+\.\d+\.\d+(?:-rc\d+)?)',contract['toolchain']).group(1)
        folder=cache/('mathlib-'+'.'.join(version.split('.')[:2]))
        roots=[folder]+[p for p in (folder/'.lake/packages').iterdir() if p.is_dir()]
        libs=[p/'.lake/build/lib' for p in roots]
        libs=[p/'lean' if (p/'lean').is_dir() else p for p in libs]
        runtime.update(lean_executable=str(cache/f'elan-home/toolchains/leanprover--lean4---v{version}/bin/lean.exe'),
                       project_dir=str(folder),lean_path=os.pathsep.join(map(str,libs)))
    return runtime
