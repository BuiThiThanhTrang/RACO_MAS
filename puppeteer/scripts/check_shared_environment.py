"""Contract/integration verification; no hosted model requests."""
import argparse
from contextlib import redirect_stdout
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sqlite3
import sys
import time
import unittest

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'tests')]
from routing_benchmark.v4.artifacts.schema import sha256_file

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=PROJECT/'runs/shared_environment_verification')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    suite=unittest.TestSuite()
    loader=unittest.TestLoader()
    for name in ('test_shared_environment','test_shared_domains','test_shared_router_protocol','test_shared_human_scoring'):
        suite.addTests(loader.loadTestsFromModule(importlib.import_module(name)))
    legacy=importlib.import_module('test_routing_benchmark_v4_artifact_execution')
    suite.addTests(unittest.FunctionTestCase(getattr(legacy,name)) for name in dir(legacy) if name.startswith('test_'))
    stream=io.StringIO()
    started=time.perf_counter()
    with redirect_stdout(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (args.output/'tests.log').write_text(stream.getvalue(),encoding='utf-8')
    report={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'elapsed_seconds':time.perf_counter()-started,'python':sys.version,'sqlite_version':sqlite3.sqlite_version,
        'platform':platform.platform(),'dependencies':{name:importlib.metadata.version(name) for name in ('openai','httpx','jsonschema','PyYAML','python-dotenv')},
        'test_hashes':{p.name:sha256_file(p) for p in (PROJECT/'tests').glob('test_shared*.py')}}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not result.wasSuccessful():
        print(stream.getvalue()[-7000:])
        raise SystemExit(1)

if __name__=='__main__': main()
