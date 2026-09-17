"""Offline policy sensitivity on GPU probe reports and an existing checkpoint."""
import argparse
import json
from pathlib import Path


def probabilities(states, weights):
    import torch
    import torch.nn.functional as F
    x = torch.tensor(states, dtype=torch.float32)
    for index in range(1, 5):
        x = F.linear(x, weights[f"fc{index}.weight"].float(), weights[f"fc{index}.bias"].float())
        if index < 4:
            x = F.relu(x)
    return torch.softmax(x, dim=1)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    ref = json.loads(Path(args.reference).read_text())
    quant = json.loads(Path(args.candidate).read_text())
    if ref["fingerprint"] != quant["fingerprint"] or [r["id"] for r in ref["rows"]] != [r["id"] for r in quant["rows"]]:
        raise ValueError("Probe reports do not match")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["input_dim"] != 8192:
        raise ValueError("Expected an 8192-dimensional policy checkpoint")
    with torch.inference_mode():
        a, b = [probabilities([r["state"] for r in report["rows"]], checkpoint["model_state_dict"])
                for report in (ref, quant)]
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Policy returned non-finite probabilities")
    print(json.dumps({"cases": len(ref["rows"]),
        "top1_agreement": (a.argmax(1) == b.argmax(1)).float().mean().item(),
        "max_probability_change": (a-b).abs().max().item(),
        "mean_total_variation": ((a-b).abs().sum(1) / 2).mean().item()}, indent=2))


if __name__ == "__main__":
    main()
