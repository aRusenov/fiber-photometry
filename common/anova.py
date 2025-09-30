import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from itertools import product

def prepare_bootstrap_anova(aucs_a, aucs_b, windows, group_labels=("A", "B")):
    """Convert bootstrapped AUCs into long-form dataframe for two-way ANOVA."""
    records = []
    for w in windows:
        for val in aucs_a[w]:
            records.append({"value": val, "group": group_labels[0], "window": str(w)})
        for val in aucs_b[w]:
            records.append({"value": val, "group": group_labels[1], "window": str(w)})
    return pd.DataFrame(records)

def run_two_way_anova(df):
    """Run two-way ANOVA using statsmodels (OLS)."""
    model = ols("value ~ C(group) * C(window)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table, model

def posthoc_pairwise(df, factor="group", alpha=0.05, method="holm"):
    """
    Run post-hoc pairwise comparisons on a single factor.
    Options:
      - method="holm" (pairwise t-tests + Holm correction)
      - method="tukey" (Tukey HSD)
    """
    if method == "tukey":
        mc = pairwise_tukeyhsd(df["value"], df[factor], alpha=alpha)
        return mc.summary()
    else:
        # Manual pairwise t-tests with Holm correction
        from scipy.stats import ttest_ind
        levels = df[factor].unique()
        pairs = list(product(levels, repeat=2))
        results = []
        p_vals = []
        for i, j in [(a,b) for a in levels for b in levels if a < b]:
            vals_i = df.loc[df[factor] == i, "value"]
            vals_j = df.loc[df[factor] == j, "value"]
            tstat, pval = ttest_ind(vals_i, vals_j)
            results.append((i, j, tstat, pval))
            p_vals.append(pval)
        reject, p_adj, _, _ = multipletests(p_vals, method=method, alpha=alpha)
        posthoc_df = pd.DataFrame({
            "comparison": [f"{i} vs {j}" for i,j,_,_ in results],
            "t_stat": [r[2] for r in results],
            "p_val": [r[3] for r in results],
            "p_adj": p_adj,
            "reject": reject
        })
        return posthoc_df