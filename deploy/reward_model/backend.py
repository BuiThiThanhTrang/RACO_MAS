"""Q5 experiment: two serial forward passes, one loaded copy of the weights.

Private binding APIs are intentionally isolated here and pinned to 0.3.16.
The embedding pass avoids assuming logits are available in embedding mode.
GPU equivalence to the Transformers reference MUST pass probe.py before serving.
"""
import logging
import threading
import time
from importlib.metadata import version

from .artifacts import LOCK, model_path, tokenizer_path, require_download
from .schemas import DIM, MAX_TOKENS, InputTooLong, make_response

logger = logging.getLogger(__name__)


class RewardBackend:
    def __init__(self, root):
        import llama_cpp
        import llama_cpp.llama_cpp as api
        from transformers import AutoTokenizer
        require_download(root)
        if version("llama-cpp-python") != LOCK["llama_cpp_python"]:
            raise RuntimeError("Unexpected llama-cpp-python version")
        if not api.llama_supports_gpu_offload():
            raise RuntimeError("llama.cpp was built without GPU offload")
        started = time.monotonic()
        self.api = api
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path(root), local_files_only=True)
        self.model = llama_cpp.Llama(model_path=str(model_path(root)),
            n_ctx=MAX_TOKENS, n_batch=512, n_ubatch=512,
            n_gpu_layers=-1, pooling_type=api.LLAMA_POOLING_TYPE_NONE,
            embedding=True, logits_all=False, verbose=False)
        if self.model.n_embd() != DIM:
            raise RuntimeError("Unexpected hidden dimension")
        self.lock = threading.Lock()
        logger.info("Q5 model loaded in %.1fs", time.monotonic() - started)

    def token_ids(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids or len(ids) > MAX_TOKENS:
            raise InputTooLong("Conversation must contain 1 to 4096 tokens; shorten it explicitly")
        # Fail closed if the GGUF tokenizer was converted differently.
        gguf_ids = self.model.tokenize(text.encode("utf-8"), add_bos=False, special=True)
        if ids != gguf_ids:
            raise RuntimeError("GGUF and Hugging Face tokenizer IDs differ")
        return ids

    def _forward(self, ids, embeddings):
        context, batch = self.model._ctx, self.model._batch
        context.kv_cache_clear()
        self.api.llama_set_embeddings(context.ctx, embeddings)
        self.api.llama_set_causal_attn(context.ctx, True)
        for start in range(0, len(ids), self.model.n_batch):
            tokens = ids[start:start + self.model.n_batch]
            # In embedding mode mark every token as output. Return only the final one.
            batch.set_batch(tokens, n_past=start, logits_all=embeddings)
            context.decode(batch)
        if embeddings:
            pointer = context.get_embeddings_ith(-1)
            if not pointer:
                raise RuntimeError("Backend did not expose the final token hidden state")
            return [float(pointer[i]) for i in range(DIM)]
        pointer = context.get_logits_ith(-1)
        if not pointer:
            raise RuntimeError("Backend did not expose final-token logits")
        # NVIDIA's HF conversion stores the scalar reward at vocabulary index zero.
        return float(pointer[0])

    def score(self, messages):
        with self.lock:
            started = time.monotonic()
            ids = self.token_ids(messages)
            try:
                reward = self._forward(ids, embeddings=False)
                state = self._forward(ids, embeddings=True)
                response = make_response(reward, state, len(ids), LOCK)
            finally:
                self.model._ctx.kv_cache_clear()
                self.model.reset()
            logger.info("Q5 tokens=%d inference_seconds=%.3f", len(ids), time.monotonic() - started)
            return response

    def close(self):
        self.model.close()
