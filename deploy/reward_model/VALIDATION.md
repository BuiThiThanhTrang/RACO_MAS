# Q5 runtime validation — 2026-09-14

- The Volume initially contained only `verification.json`; Q5 weights, tokenizer
  and `download.json` were missing. This caused the reported FileNotFoundError.
- Downloaded the pinned Q5 artifact and tokenizer on Modal CPU. The full GGUF
  SHA-256 and byte count passed validation; `download.json` was committed.
- Ran `probe_q5` successfully on NVIDIA A100 80GB PCIe (compute capability 8.0).
  Result: `{"cases": 10, "repeat_max_error": 0.0}`; exit code 0.
- The probe enforces matching HF/GGUF token IDs, finite reward, and a finite
  8192-element final-token hidden state. It also checks repeatability after
  intervening conversations. It does NOT establish BF16 equivalence.
- Improved missing/incomplete artifact errors to name the required download command.
- All 27 CPU regression tests passed, including four artifact-precondition tests.

Download run: https://modal.com/apps/bt299784/main/ap-scGa2rmG2VEAwr4v23A38a
Q5 probe run: https://modal.com/apps/bt299784/main/ap-b381tjyUujaSc8GDXiNzIV

Only Q5 and tokenizer were downloaded in this step. BF16 reference weights are
still missing. Before `probe_reference`, run `download --include-reference` and
wait for completion. Verification and deployment remain pending; the endpoint
verification gate is still closed. Historical records follow.

---

# Build fix validation — 2026-09-14

The CUDA image now builds successfully on Modal.

- Original image `im-tfw00nysawnBJcCmQz4QB8` failed because `CC=clang`
  referred to a compiler not installed by `build-essential`.
- After selecting GCC/G++, the next build reached the final link step but
  `llama-mtmd-cli` failed to resolve `libcuda.so.1` in the CPU image builder.
- Fixed the compiler variables explicitly and set `-DLLAVA_BUILD=OFF`.
  The text-model backend still uses `-DGGML_CUDA=ON` and CUDA architecture 80.
- Successfully built and installed `llama-cpp-python==0.3.16` in image
  `im-L1XOyhqqJ6jDicmzNLYQ9J` (build completed in 349 seconds).
- `build_check` ran on Modal CPU and returned
  `{"llama_cpp_python": "0.3.16", "cuda_library_present": true}`; exit code 0.
- The 23 CPU regression tests passed during this fix. Syntax and whitespace
  checks passed.

Run: https://modal.com/apps/bt299784/main/ap-ae3PZCb2N5XoWqA7q3hJSe

No GPU was allocated and no model weights were loaded by this check. The CUDA
container's missing-driver warning is expected for this CPU-only check; it does
not validate CUDA runtime execution. Q5/BF16 probes and policy evaluation remain
required. Changing the image source changes the verification fingerprint, so
previous probe reports must be regenerated before serving this revision.

The credential/environment statements below are historical observations from
September 12. A Modal profile is available for the September 14 build check.

---

# Validation status — 2026-09-12

## Completed locally

- `python -m unittest discover -s tests -v`: **23 tests passed**.
- Syntax compilation of the server package, remote client and modified policy files passed.
- `git diff --check` passed.
- Imported the real Modal App with **Modal SDK 1.5.5** without authentication.
- Checked the generated CLI help for `download`, `verify` and `deploy`.
- Exercised the production MLP forward/backward with a remote-adapter tensor on CPU.
- Confirmed the offline policy comparison reproduces the production MLP probabilities.
- Resolved the pinned HF commits and GGUF filename/size/SHA-256 from Hugging Face metadata.
  The 50 GB artifact itself has not been downloaded or hashed locally.

Local environment: Windows, Python 3.13, PyTorch 2.8.0+cpu. The proposed Modal
runtime is Linux/Python 3.11 with dependencies pinned in `modal_app.py`.
The SDK was installed into ignored `.cache/reward-sdk-env` for declaration/CLI
checks, not into the primary Python environment.

## Outstanding GPU and integration verification

This environment has no CUDA device, Modal profile, SDK token environment or
Proxy Token environment. No cloud deployment, GPU allocation or model download
was performed.

Still required:

1. Build the CUDA Image on Modal and download the pinned artifacts.
2. Run Q5 and BF16 probes and inspect reward/hidden-state/tokenization metrics.
3. Pass verification using thresholds appropriate for the actual workload.
4. Deploy and run the authenticated HTTP smoke test.
5. Evaluate a real policy checkpoint and a small end-to-end Puppeteer task set.
6. Measure VRAM, cold-start and warm-request latency on the chosen GPU.

CPU tests use fake HTTP/model outputs. They do not establish that the GGUF
preserves the reward head or exposes exactly the Transformers hidden state.
The server startup verification gate intentionally remains closed until GPU
probes pass. A successful probe is not a task-quality benchmark.

See [README.md](README.md) for commands and configuration.
