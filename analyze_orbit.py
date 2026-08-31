#!/usr/bin/env python3
"""
Post-hoc analysis of an orbit run. No GPU work: everything here is recomputed
from the per-view score vectors stored by score_orbit.

  python analyze_orbit.py outputs/<model>/<file>.jsonl
"""

import sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Pre-registered directions. Fixed a priori by the invariance argument and applied
# globally to every dataset -- never re-fit or sign-flipped per domain.
DIRECTION = {
    "orbit_cv": +1, "orbit_var": +1, "abs_orbit_offset": +1,
    "gram_cv": +1, "gram_erank_cv": +1, "drift": +1,
    "logdet": +1,   # legacy absolute feature, kept as the reference to beat
}


def cv(v):
    v = np.asarray(v, dtype=float)
    return v.std(ddof=1) / abs(v.mean())


def load(path):
    with open(path) as f:
        next(f)  # metadata header
        df = pd.read_json(f, lines=True)
    df["generator"] = df.img_path.str.split("/").str[0]
    return df


def auc_table(df, features):
    """Per-generator AUC, fakes vs the pooled real set of the same dataset."""
    real = df[df.label == 0.0]
    gens = sorted(df[df.label == 1.0].generator.unique())
    rows = []
    for feat in features:
        row = {"feature": feat}
        vals = []
        for g in gens:
            sub = pd.concat([df[(df.label == 1.0) & (df.generator == g)], real])
            if sub.label.nunique() < 2:
                continue
            a = roc_auc_score(sub.label, DIRECTION.get(feat, 1) * sub[feat]) * 100
            row[g] = round(a, 1)
            vals.append(a)
        row["MACRO"] = round(float(np.mean(vals)), 1) if vals else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


def m_ablation(df, m_values=(4, 8, 12)):
    """Free ablation: prefixes of the stored orbit are valid smaller orbits.
    t1/t2 alternate by parity, so even-length prefixes stay recipe-balanced and
    the settings are nested (no augmentation-draw noise between them)."""
    out = {}
    for m in m_values:
        assert m % 2 == 0, "use even m so the t1/t2 prefix stays balanced"
        df[f"orbit_cv_m{m}"] = df.jepa_per_view.apply(lambda v: cv(v[1:1 + m]))
        df[f"gram_cv_m{m}"] = df.gram_per_view.apply(lambda v: cv(v[1:1 + m]))
        out[m] = auc_table(df, [f"orbit_cv_m{m}", f"gram_cv_m{m}"])
    return out


def quartile_signflip(df, feature="orbit_cv"):
    """The sign-flip test. The absolute score inverts direction across content
    complexity (per-quartile AUC 0.96 -> 0.16); an orbit-relative feature should
    stay above 50 in every quartile."""
    real = df[df.label == 0.0]
    if real.empty:
        return None
    edges = np.quantile(real.logdet, [0, .25, .5, .75, 1.0])
    rows = []
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        sub = df[(df.logdet >= lo) & (df.logdet <= hi)]
        if sub.label.nunique() < 2:
            continue
        rows.append({
            "quartile": f"Q{i+1}",
            "n": len(sub),
            "logdet_auc": round(roc_auc_score(sub.label, sub.logdet) * 100, 1),
            f"{feature}_auc": round(roc_auc_score(sub.label, DIRECTION[feature] * sub[feature]) * 100, 1),
        })
    return pd.DataFrame(rows).set_index("quartile")


if __name__ == "__main__":
    df = load(sys.argv[1])
    print(f"{len(df)} images | generators: {sorted(df.generator.unique())}\n")

    feats = [f for f in ["orbit_cv", "orbit_var", "abs_orbit_offset",
                         "gram_cv", "gram_erank_cv", "drift", "logdet"] if f in df]
    print("=== AUC by feature (single global direction, no per-domain calibration) ===")
    print(auc_table(df, feats).to_string(), "\n")

    print("=== m-ablation (recomputed from stored per-view scores) ===")
    for m, tbl in m_ablation(df).items():
        print(f"-- m={m}")
        print(tbl.to_string())
    print()

    print("=== sign-flip test: AUC within content-complexity quartiles ===")
    q = quartile_signflip(df)
    print(q.to_string() if q is not None else "no reals in this file")
