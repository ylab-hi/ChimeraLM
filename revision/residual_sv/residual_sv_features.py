"""Features of residual unsupported SV calls after ChimeraLM (R3.Q6 / R3.Q11).

Inputs: Truvari output dirs (fp.vcf.gz = unsupported, tp-comp.vcf.gz = supported) for the
WGA + ChimeraLM call sets (Fig. 3b right = PromethION, Fig. 3c right = MinION; SUPPORT >= 3).
Reports SVTYPE / size / SUPPORT distributions of unsupported calls, the same for supported
calls, and for SUPPORT thresholds the fraction of unsupported vs supported calls removed.

Usage: python residual_sv_features.py <label>=<truvari_output_dir> [...] --out out_dir
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd

SIZE_BINS = [(50, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 10000), (10000, 50000), (50000, 10**12)]  # [lo, hi)
SIZE_LABELS = ["50-100 bp", "100-500 bp", "500 bp-1 kb", "1-5 kb", "5-10 kb", "10-50 kb", ">50 kb"]
SUP_BINS = [(3, 3), (4, 4), (5, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]
SUP_LABELS = ["3", "4", "5", "6-10", "11-20", "21-50", ">50"]
THRESHOLDS = [3, 5, 10, 20]
# Magic value constants
SIZE_THRESHOLD_500 = 500
SUPPORT_THRESHOLD_5 = 5


def read_vcf(path: Path) -> pd.DataFrame:
    """Read a gzipped VCF file and extract SV information.

    Args:
        path: Path to the gzipped VCF file.

    Returns:
        DataFrame with columns: chrom, pos, svtype, svlen, support.

    """
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            info = dict(kv.split("=", 1) if "=" in kv else (kv, True) for kv in f[7].split(";"))
            svtype = info.get("SVTYPE", "NA")
            if svtype == "BND":
                svtype = "TRA"
            svlen = abs(int(info.get("SVLEN", 0))) if str(info.get("SVLEN", "0")).lstrip("-").isdigit() else 0
            sup = int(info.get("SUPPORT", 0))
            rows.append((f[0], int(f[1]), svtype, svlen, sup))
    return pd.DataFrame(rows, columns=["chrom", "pos", "svtype", "svlen", "support"])


def binned(series: pd.Series, bins, labels, *, closed: bool = False) -> pd.Series:
    """Count values per bin; bins are [lo, hi) unless closed=True (integer ranges [lo, hi])."""
    out = pd.Series(0, index=labels, dtype=int)
    for (lo, hi), lab in zip(bins, labels, strict=True):
        out[lab] = int(((series >= lo) & ((series <= hi) if closed else (series < hi))).sum())
    return out


def main() -> None:
    """Analyze residual unsupported SV calls after ChimeraLM filtering.

    Generates a summary markdown report and a TSV table of SV features
    (type, size, support distributions) for unsupported and supported calls.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="label=truvari_output_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Residual unsupported SV calls after ChimeraLM\n"]
    tables = []
    for spec in args.sets:
        label, d = spec.split("=", 1)
        fp = read_vcf(Path(d) / "fp.vcf.gz")
        tp = read_vcf(Path(d) / "tp-comp.vcf.gz")
        md.append(f"## {label}: unsupported n = {len(fp):,}; supported n = {len(tp):,}\n")
        for name, df in (("unsupported", fp), ("supported", tp)):
            t = df["svtype"].value_counts()
            md.append(f"- {name} SV type: " + ", ".join(f"{k} {v:,} ({v/len(df)*100:.1f}%)" for k, v in t.items()))
            md.append(f"- {name} size: median {int(df['svlen'].median()):,} bp, <{SIZE_THRESHOLD_500} bp {(df['svlen']<SIZE_THRESHOLD_500).mean()*100:.1f}%")
            md.append(f"- {name} SUPPORT: median {int(df['support'].median())}, ≤{SUPPORT_THRESHOLD_5} reads {(df['support']<=SUPPORT_THRESHOLD_5).mean()*100:.1f}%")
            for col, bins, labels, kind in (("svlen", SIZE_BINS, SIZE_LABELS, "size"), ("support", SUP_BINS, SUP_LABELS, "support")):
                b = binned(df[col], bins, labels, closed=(kind == "support"))
                tables.append(pd.DataFrame({"set": label, "class": name, "feature": kind, "bin": b.index, "count": b.to_numpy(), "pct": b.to_numpy() / len(df) * 100}))
            tables.append(pd.DataFrame({"set": label, "class": name, "feature": "svtype", "bin": t.index, "count": t.to_numpy(), "pct": t.to_numpy() / len(df) * 100}))
        md.append("\n| SUPPORT ≥ | unsupported kept | unsupported removed (%) | supported kept | supported removed (%) | unsupported:supported |\n|---|---|---|---|---|---|")
        for thr in THRESHOLDS:
            fk, tk = int((fp["support"] >= thr).sum()), int((tp["support"] >= thr).sum())
            md.append(f"| {thr} | {fk:,} | {(1-fk/len(fp))*100:.1f} | {tk:,} | {(1-tk/len(tp))*100:.1f} | {fk/tk:.2f} : 1 |")
            tables.append(pd.DataFrame({"set": [label], "class": ["threshold"], "feature": ["support_ge"], "bin": [str(thr)], "count": [fk], "pct": [tk]}))
        md.append("")
    pd.concat(tables).to_csv(out / "residual_sv_tables.tsv", sep="\t", index=False)
    (out / "summary.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
