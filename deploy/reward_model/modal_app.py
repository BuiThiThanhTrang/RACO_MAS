"""Run all commands from repository root; see README.md."""
import logging
from pathlib import Path
import modal

from .artifacts import atomic_json, require_verified

ROOT = "/models"
app = modal.App("puppeteer-nemotron-q5")
volume = modal.Volume.from_name("puppeteer-nemotron-q5", create_if_missing=True)
package_dir = Path(__file__).resolve().parent
cpu_image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==0.34.4", "numpy==1.26.4")
    .add_local_python_source("deploy")
    .add_local_file(package_dir / "model.lock.json", "/root/deploy/reward_model/model.lock.json")
    .add_local_file(package_dir / "probe_cases.json", "/root/deploy/reward_model/probe_cases.json"))
gpu_image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "cmake", "ninja-build")
    .env({"CMAKE_ARGS": "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80", "CMAKE_BUILD_PARALLEL_LEVEL": "4"})
    .pip_install("numpy==1.26.4", "transformers==4.44.2", "huggingface_hub==0.34.4",
                 "fastapi==0.115.12", "torch==2.6.0", "accelerate==1.0.1")
    # Modal's Python image may inherit CC=clang without installing clang.
    # Set this after dependency installation to reuse the cached base layers.
    .env({
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "CUDAHOSTCXX": "/usr/bin/g++",
        "CUDACXX": "/usr/local/cuda/bin/nvcc",
        # 0.3.16 enables mtmd CLI via LLAVA_BUILD even for text-only users.
        # Its executable needs the runtime CUDA driver during the CPU image build.
        "CMAKE_ARGS": "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAVA_BUILD=OFF",
    })
    .pip_install("llama-cpp-python==0.3.16", extra_options="--no-binary=llama-cpp-python --verbose")
    .add_local_python_source("deploy")
    .add_local_file(package_dir / "model.lock.json", "/root/deploy/reward_model/model.lock.json")
    .add_local_file(package_dir / "probe_cases.json", "/root/deploy/reward_model/probe_cases.json"))


@app.function(image=gpu_image, timeout=120, max_containers=1)
def build_check():
    """Build and inspect the CUDA wheel on CPU, without loading model weights."""
    from importlib.metadata import version
    from importlib.util import find_spec
    spec = find_spec("llama_cpp")
    package = Path(next(iter(spec.submodule_search_locations)))
    if not (package / "lib" / "libggml-cuda.so").is_file():
        raise RuntimeError("Built wheel is missing its CUDA backend library")
    print({"llama_cpp_python": version("llama-cpp-python"), "cuda_library_present": True})


@app.function(image=cpu_image, volumes={ROOT: volume}, timeout=14400, max_containers=1)
def download(include_reference: bool = False):
    from .download_model import download as fetch
    fetch(ROOT, include_reference=include_reference)
    volume.commit()
    print("Pinned model artifacts downloaded and checksum verified.")


@app.function(image=gpu_image, gpu="A100-80GB", memory=65536,
              volumes={ROOT: volume}, timeout=1800, max_containers=1)
def probe_q5():
    from .probe import run_q5
    atomic_json(Path(ROOT) / "verification.json", {"passed": False})
    volume.commit()
    result = run_q5(ROOT)
    volume.commit()
    print(result)


@app.function(image=gpu_image, gpu="A100-80GB:2", memory=196608,
              volumes={ROOT: volume}, timeout=3600, max_containers=1)
def probe_reference():
    from .probe import run_reference
    atomic_json(Path(ROOT) / "verification.json", {"passed": False})
    volume.commit()
    result = run_reference(ROOT)
    volume.commit()
    print(result)


@app.function(image=cpu_image, volumes={ROOT: volume}, timeout=120, max_containers=1)
def verify(min_cosine: float, max_reward_error: float, max_norm_relative_error: float):
    import json
    from .probe import compare
    root = Path(ROOT)
    atomic_json(root / "verification.json", {"passed": False})
    volume.commit()
    report = compare(json.loads((root / "reference_probe.json").read_text()),
        json.loads((root / "q5_probe.json").read_text()), min_cosine=min_cosine,
        max_reward_error=max_reward_error, max_norm_relative_error=max_norm_relative_error)
    atomic_json(root / "verification.json", report)
    volume.commit()
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("Q5 verification failed; inspect metrics before deploying")


@app.cls(image=gpu_image, gpu="A100-80GB", memory=65536, volumes={ROOT: volume},
         timeout=600, startup_timeout=900, min_containers=0, max_containers=1, scaledown_window=300)
class RewardServer:
    @modal.enter()
    def load(self):
        from .backend import RewardBackend
        require_verified(ROOT)
        logging.basicConfig(level=logging.INFO)
        self.backend = RewardBackend(ROOT)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def score(self, payload: dict):
        from fastapi import HTTPException
        from .schemas import validate_request, InputTooLong
        try:
            messages = validate_request(payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        try:
            return self.backend.score(messages)
        except InputTooLong as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except Exception as exc:
            logging.error("Reward inference failed: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Reward inference failed") from None

    @modal.exit()
    def close(self):
        if hasattr(self, "backend"):
            self.backend.close()
