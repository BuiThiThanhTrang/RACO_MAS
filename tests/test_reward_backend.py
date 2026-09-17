import copy
import ctypes
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from deploy.reward_model.backend import RewardBackend
from deploy.reward_model.artifacts import fingerprint, require_verified, require_download, LOCK, model_path
from deploy.reward_model.probe import cases, compare
from deploy.reward_model.schemas import InputTooLong


class BackendTests(unittest.TestCase):
    def backend(self):
        backend = RewardBackend.__new__(RewardBackend)
        backend.api = Mock()
        backend.model = Mock(n_batch=512)
        backend.lock = threading.Lock()
        backend.tokenizer = Mock()
        backend.tokenizer.apply_chat_template.return_value = "formatted"
        backend.tokenizer.encode.return_value = [1, 2, 3]
        backend.model.tokenize.return_value = [1, 2, 3]
        backend.model._ctx.get_logits_ith.return_value = (ctypes.c_float * 1)(-2)
        backend.model._ctx.get_embeddings_ith.return_value = (ctypes.c_float * 8192)(*([0.1] * 8192))
        return backend

    def test_two_passes_and_cleanup(self):
        backend = self.backend()
        output = backend.score([])
        self.assertEqual(output["reward"], -2)
        self.assertEqual(len(output["last_hidden_state"]), 8192)
        flags = [c.args[1] for c in backend.api.llama_set_embeddings.call_args_list]
        self.assertEqual(flags, [False, True])
        self.assertEqual(backend.model._ctx.kv_cache_clear.call_count, 3)
        backend.model.reset.assert_called_once()

    def test_cleanup_after_decode_failure(self):
        backend = self.backend()
        backend.model._ctx.decode.side_effect = RuntimeError("GPU failure")
        with self.assertRaises(RuntimeError):
            backend.score([])
        backend.model.reset.assert_called_once()
        self.assertEqual(backend.model._ctx.kv_cache_clear.call_count, 2)

    def test_chunks_use_absolute_positions(self):
        backend = self.backend()
        backend._forward(list(range(1100)), True)
        calls = backend.model._batch.set_batch.call_args_list
        self.assertEqual([c.kwargs["n_past"] for c in calls], [0, 512, 1024])
        self.assertEqual([len(c.args[0]) for c in calls], [512, 512, 76])

    def test_tokenizer_mismatch_rejected(self):
        backend = self.backend()
        backend.model.tokenize.return_value = [9]
        with self.assertRaises(RuntimeError):
            backend.score([])
        backend.model._ctx.decode.assert_not_called()

    def test_overlength_rejected_before_decode(self):
        backend = self.backend()
        backend.tokenizer.encode.return_value = [1] * 4097
        with self.assertRaises(InputTooLong):
            backend.score([])
        backend.model._ctx.decode.assert_not_called()

    def test_missing_hidden_pointer_rejected(self):
        backend = self.backend()
        backend.model._ctx.get_embeddings_ith.return_value = None
        with self.assertRaises(RuntimeError):
            backend.score([])


class DownloadPreconditionTests(unittest.TestCase):
    def test_missing_manifest_explains_download_step(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "modal_app::download"):
                require_download(directory)

    def test_corrupt_manifest_explains_download_step(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "download.json").write_text("incomplete")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                require_download(directory)

    def test_missing_weights_rejected_even_with_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "download.json").write_text(json.dumps(LOCK))
            with self.assertRaisesRegex(RuntimeError, "modal_app::download"):
                require_download(directory)

    def test_wrong_weight_size_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "download.json").write_text(json.dumps(LOCK))
            target = model_path(directory)
            target.parent.mkdir(parents=True)
            target.write_bytes(b"partial")
            with self.assertRaisesRegex(RuntimeError, "missing or changed"):
                require_download(directory)


class VerificationTests(unittest.TestCase):
    def reports(self):
        rows = [{"id": c["id"], "token_ids": [1, 2], "state": [1.0] * 8192,
                 "reward": float(i % 2)} for i, c in enumerate(cases())]
        ref = {"fingerprint": fingerprint(), "rows": rows}
        quant = copy.deepcopy(ref)
        quant["repeat_max_error"] = 0
        return ref, quant

    def compare(self, ref, quant):
        return compare(ref, quant, min_cosine=0.99, max_reward_error=0.5, max_norm_relative_error=0.1)

    def test_matching_probes_pass(self):
        self.assertTrue(self.compare(*self.reports())["passed"])

    def test_semantic_mismatch_fails(self):
        for kind in ("tokens", "norm", "reward", "repeat", "rank"):
            ref, quant = self.reports()
            if kind == "tokens": quant["rows"][0]["token_ids"] = [5]
            if kind == "norm": quant["rows"][0]["state"] = [2.0] * 8192
            if kind == "reward": quant["rows"][0]["reward"] = 10
            if kind == "repeat": quant["repeat_max_error"] = 0.1
            if kind == "rank":
                ref["rows"][0]["reward"], ref["rows"][1]["reward"] = 0.1, 0.2
                quant["rows"][0]["reward"], quant["rows"][1]["reward"] = 0.2, 0.1
            with self.subTest(kind=kind):
                self.assertFalse(self.compare(ref, quant)["passed"])

    def test_stale_probes_and_nonfinite_state_rejected(self):
        ref, quant = self.reports()
        quant["fingerprint"] = "old"
        with self.assertRaises(ValueError): self.compare(ref, quant)
        ref, quant = self.reports()
        quant["rows"][0]["state"][0] = float("nan")
        with self.assertRaises(ValueError): self.compare(ref, quant)

    def test_endpoint_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verification.json"
            with self.assertRaises(FileNotFoundError): require_verified(directory)
            for report in ({"passed": False, "fingerprint": fingerprint()}, {"passed": True, "fingerprint": "old"}):
                path.write_text(json.dumps(report))
                with self.assertRaises(RuntimeError): require_verified(directory)
            path.write_text(json.dumps({"passed": True, "fingerprint": fingerprint()}))
            require_verified(directory)


if __name__ == "__main__":
    unittest.main()
