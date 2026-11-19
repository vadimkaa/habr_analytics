"""Additional descriptive checks agreed during final editorial review."""
import json
import math

import pandas as pd

from habrpub import RESULTS
from habrpub.analysis import corr, load_corpus, prior_density, recent


def compute():
    corpora = {c: load_corpus(c) for c in ("ai", "wide")}
    dates = pd.concat(
        [df[["id", "dt_utc"]] for df in corpora.values()], ignore_index=True
    )
    if dates.groupby("id").dt_utc.nunique().gt(1).any():
        raise ValueError("Shared articles have inconsistent publication timestamps")
    dates = dates.drop_duplicates("id").sort_values(["dt_utc", "id"]).copy()
    assert len(dates) == 46845
    dates["union_density"] = prior_density(dates.dt_utc)
    density = dates.set_index("id").union_density
    values = {}
    for c, df in corpora.items():
        d = recent(df)
        d = d[d.density_complete].copy()
        assert len(d) == {"ai": 6775, "wide": 16577}[c]
        for label, is_weekend in (("wd", False), ("we", True)):
            for period, lo, hi in (("day", 9, 15), ("eve", 16, 22)):
                subset = d[(d.is_weekend == is_weekend) & d.hour.between(lo, hi)]
                values[f"ed_{c}_{label}_{period}"] = float(subset.vpd.median())
        if c == "ai":
            eight = d[d.hour == 8]
            values["ed_ai_8_vpd"] = float(eight.vpd.median())
            values["ed_ai_8_n"] = len(eight)
        rho = corr(d.id.map(density), d.vpd)["rho"]
        values[f"ed_{c}_union_rho"] = float(rho)
    return values


def frozen():
    return json.loads((RESULTS / "19_editorial_checks.json").read_text(encoding="utf-8"))


def validate(actual, expected):
    """The v1.1.0 snapshot retained only rounded values, not raw floats.

    Check raw recomputed values against the half-unit of the stored precision;
    freeze the publication representation even at an exact rounding boundary.
    """
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if key.endswith("_n"):
            assert actual[key] == value, key
        else:
            tolerance = 0.0005 if key.endswith("_rho") else 0.05
            assert math.isclose(actual[key], value, rel_tol=0, abs_tol=tolerance + 1e-10), key


def publication_values():
    return {
        k: (str(x) if k.endswith("_n") else
            f"{x:.3f}".replace("-", "−") if k.endswith("_rho") else f"{x:.1f}")
        for k, x in frozen().items()
    }


if __name__ == "__main__":
    actual = compute()
    validate(actual, frozen())
    print(json.dumps(actual, ensure_ascii=False, indent=2))
