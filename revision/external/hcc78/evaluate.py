r"""Evaluate the frozen ChimeraLM model on an external WGA dataset (HCC78, PRJNA875576).

Inputs
  --pred   <label>=<predictions.txt>   ChimeraLM predictions (name\tclass; 1 = artifact), one per WGA sample
  --sup    <label>=<threshold_1000.sup.txt>  bulk-support labels from `annotate` (name\tsupport[\tpaths]);
                                        support 0 = no bulk match (artifact), >=1 = genuine
  --stats  <label>=<chimeric_stats.tsv.gz>  optional read lengths from chimeric_read_stats.py (for length bins)
  --flagstat <label>=<flagstat.txt>    samtools flagstat of the sample BAM (for chimeric fraction)
  --nchim  <label>=<int>               number of chimeric reads (dbp count-chimeric); used with flagstat
  --out    <dir>

Outputs summary.md and metrics.tsv: precision / recall / F1 (positive = artifact) vs bulk-support labels,
confusion counts, artifact-call rate, and chimeric-read fraction before/after filtering.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def kv(pairs: list[str] | None) -> dict[str, str]:
    """Convert list of "key=value" strings to dict."""
    return {p.split("=", 1)[0]: p.split("=", 1)[1] for p in (pairs or [])}


def prf(y, p):
    """Calculate precision, recall, F1, and confusion matrix values."""
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def mapped_primary_reads(flagstat: Path) -> int:
    """Extract mapped primary read count from samtools flagstat output."""
    txt = flagstat.read_text()
    m = re.search(r"(\d+) \+ \d+ primary mapped", txt)
    if m:
        return int(m.group(1))
    return int(re.search(r"(\d+) \+ \d+ primary\b", txt).group(1))


def main() -> None:
    """Evaluate ChimeraLM predictions against bulk-support labels and report metrics."""
    MIN_SAMPLE_SIZE = 10
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True)
    ap.add_argument("--sup", action="append")
    ap.add_argument("--stats", action="append")
    ap.add_argument("--flagstat", action="append")
    ap.add_argument("--nchim", action="append")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preds, sups, stats, flags, nchim = kv(a.pred), kv(a.sup), kv(a.stats), kv(a.flagstat), kv(a.nchim)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md, rows = ["# External evaluation (frozen ChimeraLM)\n"], []
    for label, ppath in preds.items():
        pred = pd.read_csv(ppath, sep="\t", header=None, names=["name", "pred"], dtype={"name": "string", "pred": "int8"})
        n = len(pred)
        art = int(pred["pred"].sum())
        md.append(f"## {label}\n- chimeric reads predicted: {n:,}; called artifact: {art:,} ({art/n*100:.1f}%)")
        row = {"sample": label, "n_chimeric": n, "n_artifact_called": art, "artifact_rate": art / n}
        if label in flags and label in nchim:
            mapped = mapped_primary_reads(Path(flags[label]))
            nc = int(nchim[label])
            kept = mapped - art
            md.append(f"- mapped primary reads: {mapped:,}; chimeric: {nc:,} ({nc/mapped*100:.2f}%); after ChimeraLM: {nc-art:,} chimeric of {kept:,} reads ({(nc-art)/kept*100:.2f}%)")
            row |= {"mapped": mapped, "chimeric_frac_before": nc / mapped, "chimeric_frac_after": (nc - art) / kept}
        if label in sups:
            sup = pd.read_csv(sups[label], sep="\t", header=None, usecols=[0, 1], names=["name", "support"], dtype={"name": "string", "support": "int16"})
            m = pred.merge(sup, on="name", how="inner")
            m["label"] = (m["support"] == 0).astype("int8")
            r = prf(m["label"].to_numpy(), m["pred"].to_numpy())
            md.append(f"- labelled (bulk-matched) reads: {len(m):,}; artifact (support 0): {int(m['label'].sum()):,} ({m['label'].mean()*100:.1f}%); genuine: {int((m['label']==0).sum()):,}")
            md.append(f"- precision {r['precision']:.3f}  recall {r['recall']:.3f}  F1 {r['f1']:.3f}  (TP {r['tp']:,} FP {r['fp']:,} FN {r['fn']:,} TN {r['tn']:,})")
            row |= dict(n_labelled=len(m), n_label_artifact=int(m["label"].sum()), **r)
            if label in stats:
                st = pd.read_csv(stats[label], sep="\t", usecols=["name", "read_len"], dtype={"name": "string"})
                mm = m.merge(st, on="name", how="left")
                bins = pd.cut(mm["read_len"], [0, 2000, 8000, 16000, 32768, float("inf")], labels=["≤2 k", "2-8 k", "8-16 k", "16-32 k", ">32 k"])
                md.append("\n| length bin | n | precision | recall | F1 |\n|---|---|---|---|---|")
                for b in bins.cat.categories:
                    s = mm[bins == b]
                    if len(s) < MIN_SAMPLE_SIZE:
                        continue
                    rr = prf(s["label"].to_numpy(), s["pred"].to_numpy())
                    md.append(f"| {b} | {len(s):,} | {rr['precision']:.3f} | {rr['recall']:.3f} | {rr['f1']:.3f} |")
        rows.append(row)
        md.append("")
    pd.DataFrame(rows).to_csv(out / "metrics.tsv", sep="\t", index=False)
    (out / "summary.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
