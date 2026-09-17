"""REWARD_MODEL_URL / MODAL_KEY / MODAL_SECRET must be set."""
import time
from puppeteer.reward_api.client import RewardClient


def main():
    client = RewardClient({"read_timeout_seconds": 180, "max_attempts": 1})
    try:
        for answer in ("1+1=2", "1+1=3"):
            start = time.monotonic()
            state, reward = client.score([
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": answer},
            ])
            print({"answer": answer, "reward": reward, "dimensions": len(state),
                   "seconds": round(time.monotonic() - start, 3)})
    finally:
        client.close()


if __name__ == "__main__":
    main()
