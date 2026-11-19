"""Additional descriptive checks agreed during final editorial review."""
import json

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
                values[f"ed_{c}_{label}_{period}"] = f"{subset.vpd.median():.1f}"
        if c == "ai":
            eight = d[d.hour == 8]
            values["ed_ai_8_vpd"] = f"{eight.vpd.median():.1f}"
            values["ed_ai_8_n"] = str(len(eight))
        rho = corr(d.id.map(density), d.vpd)["rho"]
        values[f"ed_{c}_union_rho"] = f"{rho:.3f}".replace("-", "−")
    return values


if __name__ == "__main__":
    values = compute()
    target = RESULTS / "19_editorial_checks.json"
    target.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(values, ensure_ascii=False, indent=2))
