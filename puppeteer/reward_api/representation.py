"""Remote adapter separate from OpenAI/Chroma initialization."""
import torch
from .client import RewardClient


class RemoteRewardRepresentation:
    dim = 8192

    def __init__(self, config, device):
        self.device = device
        self.client = RewardClient(config)

    def __call__(self, messages):
        values, reward = self.client.score(messages)
        state = torch.tensor(values, dtype=torch.float32, device=self.device).reshape(1, self.dim)
        if not torch.isfinite(state).all():
            raise ValueError("Reward state overflowed float32")
        return state, reward

    def close(self):
        self.client.close()
