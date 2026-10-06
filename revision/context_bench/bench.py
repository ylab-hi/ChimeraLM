"""Inference time / GPU memory vs input length and backbone context (R3.Q8).

Forward-only (no training). Random A/C/G/T token sequences, bf16 autocast, one GPU.
  - released ChimeraLM (hyenadna-small-32k-seqlen) at padded lengths 2k..32k, batch 12 (CLI default) and 1
  - hyenadna-medium-160k-seqlen backbone with the same (randomly initialised) head at 32k and 160k
Reports reads/s and peak allocated GPU memory. Parameter counts included.

Usage: uv run --no-sync python revision/context_bench/bench.py --out revision/context_bench/out_<date>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import chimeralm.models
from chimeralm.models.components import hyena

TOKENS = torch.tensor([7, 8, 9, 10])  # A C G T


def make_batch(batch: int, length: int, device) -> torch.Tensor:
    return TOKENS[torch.randint(0, 4, (batch, length))].to(device)


def bench(net: torch.nn.Module, batch: int, length: int, device, n_iter: int = 10) -> dict:
    net.eval()
    x = make_batch(batch, length, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):  # warm-up
            net(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            net(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_iter
    return {
        "batch": batch,
        "length": length,
        "sec_per_batch": dt,
        "reads_per_sec": batch / dt,
        "peak_mem_GB": torch.cuda.max_memory_allocated(device) / 1e9,
    }


def try_bench(net, batch, length, device):
    try:
        return bench(net, batch, length, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"batch": batch, "length": length, "oom": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    results = {"gpu": torch.cuda.get_device_name(device), "runs": []}

    model = chimeralm.models.ChimeraLM.from_pretrained("yangliz5/chimeralm").to(device)
    net = model.net
    n_params = sum(p.numel() for p in net.parameters())
    print(f"released model params: {n_params:,}")
    for batch in (12, 1):
        for length in (2_048, 4_096, 8_192, 16_384, 32_768):
            r = try_bench(net, batch, length, device) | {"model": "hyenadna-small-32k (released ChimeraLM)", "params": n_params}
            results["runs"].append(r)
            print(r, flush=True)
    del model, net
    torch.cuda.empty_cache()

    net160 = hyena.HyenaDna(
        number_of_classes=2,
        backbone_name="hyenadna-medium-160k-seqlen",
        head=hyena.BinarySequenceClassifier(input_dim=256, hidden_dim=512, num_layers=2, dropout=0.1,
                                            pooling_type="attention", activation="gelu", use_residual=True),
    ).to(device)
    n160 = sum(p.numel() for p in net160.parameters())
    print(f"160k backbone params: {n160:,}")
    for batch, length in ((12, 32_768), (1, 32_768), (12, 160_000), (4, 160_000), (1, 160_000)):
        r = try_bench(net160, batch, length, device) | {"model": "hyenadna-medium-160k (untrained head)", "params": n160}
        results["runs"].append(r)
        print(r, flush=True)

    (out / "bench.json").write_text(json.dumps(results, indent=2))
    with open(out / "bench.md", "w") as fh:
        fh.write("| model | batch | input length | s/batch | reads/s | peak GPU mem (GB) |\n|---|---|---|---|---|---|\n")
        for r in results["runs"]:
            if r.get("oom"):
                fh.write(f"| {r['model']} | {r['batch']} | {r['length']:,} | OOM | OOM | >80 |\n")
            else:
                fh.write(f"| {r['model']} | {r['batch']} | {r['length']:,} | {r['sec_per_batch']:.3f} | {r['reads_per_sec']:.1f} | {r['peak_mem_GB']:.2f} |\n")
    print(open(out / "bench.md").read())


if __name__ == "__main__":
    main()
