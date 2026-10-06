"""Read-length / 32 k-truncation analysis for revision R1 (R1.Q2, R1.Q3, R1.Q4, R3.Q8, R3.Q13).

Inputs
  --stats  <platform>=<tsv.gz>   output of chimeric_read_stats.py (one per platform)
  --pred   <platform>=<txt>      ChimeraLM predictions "name\tclass" (1 = artifact)
  --test   <tsv.gz>              output of test_set_ids.py (name, label, seq_len)
  --test-platform <platform>     platform whose predictions cover the test split (p2)
  --out    <dir>                 writes summary.md, tables (*.tsv) and figure (pdf/png)

Usage
  uv run --no-sync python revision/read_length/analyze.py \
      --stats p2=out/p2_chimeric_stats.tsv.gz --stats mk1c=out/mk1c_chimeric_stats.tsv.gz \
      --pred p2=.../hyena_p2_765108_bulk_p2_predicts.txt --pred mk1c=.../hyena_p2_765108_bulk_mk1c_predicts.txt \
      --test out/test_ids.tsv.gz --test-platform p2 --out out/analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAX_LEN = 32_768
BINS = [0, 2_000, 8_000, 16_000, MAX_LEN, np.inf]
BIN_LABELS = ["≤2 k", "2–8 k", "8–16 k", "16–32 k", ">32 k"]
PLATFORM_NAME = {"p2": "PromethION", "mk1c": "MinION"}
C_RETAIN, C_REMOVE = "#1b7f79", "#d1495b"
C_PLATFORM = {"p2": "#1b7f79", "mk1c": "#e9a03b"}


def n50(x: np.ndarray) -> int:
    s = np.sort(x)[::-1]
    c = np.cumsum(s)
    return int(s[np.searchsorted(c, c[-1] / 2)])


def kv(pairs: list[str]) -> dict[str, Path]:
    return {p.split("=", 1)[0]: Path(p.split("=", 1)[1]) for p in pairs}


def junctions_from_segments(segments: str) -> tuple[int, int]:
    """First/last junction (midpoint between consecutive sorted segments) in read coordinates."""
    iv = [tuple(map(int, s.split("-"))) for s in segments.split(";")]
    j = [(iv[i][1] + iv[i + 1][0]) // 2 for i in range(len(iv) - 1)]
    return (min(j), max(j)) if j else (-1, -1)


def load_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["name", "read_len", "mapq", "n_segments", "segments"],
        dtype={"name": "string", "read_len": "int64", "mapq": "int16", "n_segments": "int32", "segments": "string"},
    )
    # recompute junctions here so TSVs produced before the junction-definition fix are usable
    fj, lj = zip(*(junctions_from_segments(s) for s in df["segments"].to_numpy()), strict=True)
    df["first_junction"] = np.asarray(fj, dtype="int64")
    df["last_junction"] = np.asarray(lj, dtype="int64")
    return df.drop(columns="segments")


def load_pred(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=["name", "pred"], dtype={"name": "string", "pred": "int8"})


def prf(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float, int, int, int, int]:
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return prec, rec, f1, tp, fp, fn, tn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="append", required=True)
    ap.add_argument("--pred", action="append", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--test-platform", default="p2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats_paths, pred_paths = kv(args.stats), kv(args.pred)
    md: list[str] = ["# Read-length / truncation analysis\n"]

    data: dict[str, pd.DataFrame] = {}
    length_rows, bin_rows = [], []
    for plat, spath in stats_paths.items():
        df = load_stats(spath)
        pred = load_pred(pred_paths[plat])
        n_stats, n_pred = len(df), len(pred)
        df = df.merge(pred, on="name", how="left")
        n_matched = int(df["pred"].notna().sum())
        df["pred"] = df["pred"].fillna(-1).astype("int8")
        df["bin"] = pd.cut(df["read_len"], BINS, labels=BIN_LABELS, right=True)
        data[plat] = df

        L = df["read_len"].to_numpy()
        over = df["read_len"] > MAX_LEN
        blind = over & (df["first_junction"] >= MAX_LEN)
        partial = over & (df["first_junction"] < MAX_LEN) & (df["last_junction"] >= MAX_LEN)
        md.append(f"## {PLATFORM_NAME[plat]} ({plat}) WGA chimeric reads\n")
        md.append(f"- chimeric reads in BAM: {n_stats:,}; predictions: {n_pred:,}; matched: {n_matched:,}")
        md.append(f"- read length: min {L.min():,}, median {int(np.median(L)):,}, mean {L.mean():,.0f}, N50 {n50(L):,}, max {L.max():,}")
        md.append(f"- reads > {MAX_LEN:,} bp: {over.sum():,} ({over.mean()*100:.3f}%)")
        md.append(f"  - of which ALL junctions lie beyond {MAX_LEN:,} bp (model window junction-free): {blind.sum():,} ({blind.sum()/max(over.sum(),1)*100:.1f}% of >32 k; {blind.mean()*100:.4f}% of all chimeric reads)")
        md.append(f"  - of which SOME but not all junctions lie beyond the window: {partial.sum():,} ({partial.sum()/max(over.sum(),1)*100:.1f}% of >32 k)")
        for lo, hi in [(0, 2_000), (2_000, 8_000), (8_000, 16_000), (16_000, MAX_LEN), (MAX_LEN, np.inf)]:
            m = (df["read_len"] > lo) & (df["read_len"] <= hi)
            length_rows.append({"platform": plat, "bin": f"{lo}-{hi}", "n": int(m.sum()), "frac": float(m.mean())})

        # artifact-call rate and segments per bin
        g = df[df["pred"] >= 0].groupby("bin", observed=False)
        tbl = g.agg(n=("pred", "size"), artifact_rate=("pred", "mean"), mean_segments=("n_segments", "mean"),
                    median_len=("read_len", "median"))
        tbl["platform"] = plat
        bin_rows.append(tbl.reset_index())
        md.append("\n| length bin | n | % called artifact | mean #segments |\n|---|---|---|---|")
        for b, r in tbl.iterrows():
            md.append(f"| {b} | {int(r['n']):,} | {r['artifact_rate']*100:.1f} | {r['mean_segments']:.2f} |")

        # retained vs removed (R1.Q4)
        keep = df.loc[df["pred"] == 0, "read_len"].to_numpy()
        drop = df.loc[df["pred"] == 1, "read_len"].to_numpy()
        md.append(f"\nRetained (pred 0): n {len(keep):,}, median {int(np.median(keep)):,}, mean {keep.mean():,.0f}, N50 {n50(keep):,}, >32 k {(keep>MAX_LEN).mean()*100:.3f}%")
        md.append(f"Removed  (pred 1): n {len(drop):,}, median {int(np.median(drop)):,}, mean {drop.mean():,.0f}, N50 {n50(drop):,}, >32 k {(drop>MAX_LEN).mean()*100:.3f}%\n")

    pd.DataFrame(length_rows).to_csv(out / "length_bins.tsv", sep="\t", index=False)
    pd.concat(bin_rows).to_csv(out / "artifact_rate_by_bin.tsv", sep="\t", index=False)

    # ---- test set stratified metrics ----
    test = pd.read_csv(args.test, sep="\t", dtype={"name": "string", "label": "int8", "seq_len": "int64"})
    tp_plat = args.test_platform
    merged = test.merge(data[tp_plat][["name", "pred", "read_len", "first_junction"]], on="name", how="left")
    covered = merged["pred"].notna() & (merged["pred"] >= 0)
    md.append(f"## Held-out test split ({len(test):,} reads), predictions from {PLATFORM_NAME[tp_plat]} WGA run\n")
    md.append(f"- covered by WGA predictions: {covered.sum():,} ({covered.mean()*100:.1f}%); label 1 (artifact) among covered: {int(merged.loc[covered,'label'].sum()):,}")
    md.append(f"- uncovered (bulk-sampled genuine reads, not in WGA BAM): {(~covered).sum():,}, of which label 0: {int((merged.loc[~covered,'label']==0).sum()):,}")
    mt = merged[covered].copy()
    mt["pred"] = mt["pred"].astype("int8")
    mismatch = (mt["seq_len"] != mt["read_len"]).sum()
    md.append(f"- seq_len (parquet) vs read_len (BAM) mismatches: {mismatch:,}")
    mt["bin"] = pd.cut(mt["seq_len"], BINS, labels=BIN_LABELS, right=True)
    rows = []
    p, r, f, tp, fp, fn, tn = prf(mt["label"].to_numpy(), mt["pred"].to_numpy())
    rows.append({"bin": "all", "n": len(mt), "n_artifact": int(mt["label"].sum()), "precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    for b in BIN_LABELS:
        s = mt[mt["bin"] == b]
        if len(s) == 0:
            continue
        p, r, f, tp, fp, fn, tn = prf(s["label"].to_numpy(), s["pred"].to_numpy())
        rows.append({"bin": b, "n": len(s), "n_artifact": int(s["label"].sum()), "precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    met = pd.DataFrame(rows)
    met.to_csv(out / "test_metrics_by_bin.tsv", sep="\t", index=False)
    md.append("\n| bin | n | n artifact | precision | recall | F1 |\n|---|---|---|---|---|---|")
    for _, r in met.iterrows():
        md.append(f"| {r['bin']} | {int(r['n']):,} | {int(r['n_artifact']):,} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    over_t = mt["seq_len"] > MAX_LEN
    if over_t.any():
        blind_t = over_t & (mt["first_junction"] >= MAX_LEN)
        md.append(f"\n- test reads > 32 k: {over_t.sum():,}; junction-free window: {blind_t.sum():,}")

    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    edges = np.logspace(np.log10(50), np.log10(300_000), 80)
    for plat, df in data.items():
        ax.hist(df["read_len"], bins=edges, histtype="step", lw=1.6, color=C_PLATFORM[plat],
                label=f"WGA {PLATFORM_NAME[plat]} (n = {len(df):,})", density=True)
    ax.axvline(MAX_LEN, color="k", ls="--", lw=1)
    ax.text(MAX_LEN * 1.1, ax.get_ylim()[1] * 0.9, "32,768 bp", fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Chimeric read length (bp)")
    ax.set_ylabel("Density")
    ax.set_title("a  Length of chimeric reads", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    w = 0.38
    x = np.arange(len(BIN_LABELS))
    for i, (plat, df) in enumerate(data.items()):
        g = df[df["pred"] >= 0].groupby("bin", observed=False)["pred"]
        rate = g.mean().reindex(BIN_LABELS).to_numpy() * 100
        n = g.size().reindex(BIN_LABELS).to_numpy()
        bars = ax.bar(x + (i - 0.5) * w, rate, w, color=C_PLATFORM[plat], label=f"WGA {PLATFORM_NAME[plat]}")
        for b, nn in zip(bars, n, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{int(nn):,}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(x, BIN_LABELS)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Chimeric reads called artifact (%)")
    ax.set_xlabel("Read length bin (bp)")
    ax.set_title("b  Artifact call rate by read length", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[1, 0]
    mb = met[met["bin"] != "all"]
    xb = np.arange(len(mb))
    for j, (col, c) in enumerate([("precision", "#4c72b0"), ("recall", "#dd8452"), ("f1", "#55a868")]):
        ax.bar(xb + (j - 1) * 0.27, mb[col], 0.27, color=c, label=col.capitalize() if col != "f1" else "F1")
    for xi, nn in zip(xb, mb["n"], strict=True):
        ax.text(xi, 1.02, f"n = {int(nn):,}", ha="center", fontsize=6.5)
    ax.set_xticks(xb, mb["bin"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (held-out test set)")
    ax.set_xlabel("Read length bin (bp)")
    ax.set_title("c  Test-set performance by read length", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")

    ax = axes[1, 1]
    for plat, df in data.items():
        ls = "-" if plat == "p2" else "--"
        ax.hist(df.loc[df["pred"] == 0, "read_len"], bins=edges, histtype="step", lw=1.5, ls=ls, color=C_RETAIN,
                density=True, label=f"{PLATFORM_NAME[plat]} retained (genuine)")
        ax.hist(df.loc[df["pred"] == 1, "read_len"], bins=edges, histtype="step", lw=1.5, ls=ls, color=C_REMOVE,
                density=True, label=f"{PLATFORM_NAME[plat]} removed (artifact)")
    ax.axvline(MAX_LEN, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Chimeric read length (bp)")
    ax.set_ylabel("Density")
    ax.set_title("d  Retained vs removed chimeric reads", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5)

    for a in axes.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "sf_read_length.pdf")
    fig.savefig(out / "sf_read_length.png", dpi=200)
    print(f"figure -> {out / 'sf_read_length.pdf'}")


if __name__ == "__main__":
    main()
