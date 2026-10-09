r"""Score per-read predictions on the held-out test split (R3.Q9): precision / recall / F1, artifact = positive.

Usage: python score_test.py --labels test_ids.tsv.gz  name=predictions.txt [name=predictions.txt ...]
  test_ids.tsv.gz: name, label (1 = artifact), seq_len  (from revision/read_length/test_set_ids.py)
  predictions.txt: name\tclass (1 = artifact), as written by PredictionWriter
"""

import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def prf(y, p):
    """Compute precision, recall, F1, and confusion matrix elements."""
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    pr = tp / (tp + fp)
    rc = tp / (tp + fn)
    f1 = 2 * pr * rc / (pr + rc)
    return pr, rc, f1, tp, fp, fn, tn


ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("preds", nargs="+")
a = ap.parse_args()
lab = pd.read_csv(a.labels, sep="\t", dtype={"name": "string", "label": "int8"})[["name", "label"]]
logger.info("| model | n | TP | FP | FN | TN | precision | recall | F1 |\n|---|---|---|---|---|---|---|---|---|")
for spec in a.preds:
    name, path = spec.split("=", 1)
    pred = pd.read_csv(path, sep="\t", header=None, names=["name", "pred"], dtype={"name": "string", "pred": "int8"})
    m = lab.merge(pred, on="name", how="inner")
    if len(m) != len(lab):
        msg = f"{name}: matched {len(m)} of {len(lab)} test reads"
        raise ValueError(msg)
    pr, rc, f1, tp, fp, fn, tn = prf(m["label"].to_numpy(), m["pred"].to_numpy())
    logger.info(f"| {name} | {len(m):,} | {tp:,} | {fp:,} | {fn:,} | {tn:,} | {pr:.3f} | {rc:.3f} | {f1:.3f} |")
