"""Exploratory follow-up to the already observed 08:00 peak.

The two corpora are not independent; the exclusive wide subset is an additional
descriptive check. The year window and VPD are the same as in stage 19.
No inference here establishes a causal effect or a best hour.
"""
import json

import numpy as np
import pandas as pd

from habrpub.analysis import bootstrap_cluster, load_corpus, recent


def median_or_none(series):
    return float(series.median()) if len(series) else None


def summary(d):
    return {
        str(hour): {
            "n": int(len(g)),
            "n_authors": int(g.author_key.nunique()),
            "corp_share": float(g.is_corp.mean()) if len(g) else None,
            "vpd": median_or_none(g.vpd),
            "month_rank": median_or_none(g.month_rank),
        }
        for hour in (7, 8, 9)
        for g in [d[d.hour == hour]]
    }


def ratio(z):
    center = z.loc[z.hour == 8, "vpd"].median()
    neighbors = z.loc[z.hour.isin([7, 9]), "vpd"].median()
    return center / neighbors if np.isfinite(neighbors) and neighbors > 0 else np.nan


def diagnostic(d):
    d = d.copy()
    d["month_rank"] = d.groupby("month_id").views.rank(pct=True)
    morning = d[d.hour.isin([7, 8, 9])]
    eight = morning[morning.hour == 8]
    sections = {
        "all": morning,
        "personal": morning[~morning.is_corp],
        "corporate": morning[morning.is_corp],
        "weekday": morning[~morning.is_weekend],
        "weekend": morning[morning.is_weekend],
        "weekday_personal": morning[~morning.is_weekend & ~morning.is_corp],
        "weekend_personal": morning[morning.is_weekend & ~morning.is_corp],
    }
    # Fixed midpoint of the nominal year window, chosen before this check.
    cut = pd.Timestamp("2026-05-09", tz="Europe/Moscow")
    sections["early"] = morning[morning.dt_msk < cut]
    sections["late"] = morning[morning.dt_msk >= cut]
    out = {
        "n_year": int(len(d)),
        "hours": {name: summary(frame) for name, frame in sections.items()},
        "median_8_ci95_author": bootstrap_cluster(eight, lambda z: z.vpd.median()),
        "ratio_8_to_7_9": {},
    }
    for name in ("all", "personal", "weekday", "early", "late"):
        frame = sections[name]
        if frame.hour.eq(8).sum() >= 20 and frame.hour.ne(8).sum() >= 20:
            out["ratio_8_to_7_9"][name] = {
                "ratio": float(ratio(frame)),
                "ci95_author": bootstrap_cluster(frame, ratio),
            }
    return out


def compute():
    corpora = {c: load_corpus(c) for c in ("ai", "wide")}
    year = {c: recent(d).loc[lambda x: x.density_complete].copy()
            for c, d in corpora.items()}
    result = {c: diagnostic(d) for c, d in year.items()}
    result["wide_excluding_ai"] = diagnostic(
        year["wide"][~year["wide"].id.isin(corpora["ai"].id)]
    )
    result["weekend_not_above_weekday_median"] = {
        c: float((d.loc[d.is_weekend, "vpd"] <= d.loc[~d.is_weekend, "vpd"].median()).mean())
        for c, d in year.items()
    }
    return result


if __name__ == "__main__":
    print(json.dumps(compute(), ensure_ascii=False, allow_nan=False, indent=2))
