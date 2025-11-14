import pandas as pd
import statsmodels.formula.api as smf

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("../licks_long_for_SPSS.csv")

# Ensure correct types
df["group"] = df["group"].astype("category")
df["condition"] = df["condition"].astype("category")
df["mouse"] = df["mouse"].astype("category")
df = df.sort_values(["mouse", "condition", "group", "time_min"])

# Compute lick rate
df["lick_rate"] = df.groupby(["mouse", "condition", "group"])["cum_lick_pct"].diff().fillna(df["cum_lick_pct"])

# Quadratic term for curvature (optional)
df["time_min_sq"] = df["time_min"] ** 2

# Exclude specific mice
exclude = ["656_3", "201_3", "653_5", "201_2", "662_4"]

# Container for results
results_list = []

# -----------------------------
# Loop over experimental groups
# -----------------------------
for grp in df["group"].unique():
    print(f"\n=== Group: {grp} ===")
    df_grp = df[df["group"] == grp].copy()
    df_grp = df_grp[~df_grp['mouse'].isin(exclude)]
    mice = df_grp['mouse'].unique()
    print(f"Mice (N={len(mice)}: {mice}")

    # ----- Choose model here -----
    # Quadratic version
    # formula = "cum_lick_pct ~ time_min + time_min_sq + condition * laser + time_min:condition + time_min:laser + time_min_sq:condition + time_min_sq:laser"

    # Simpler version
    formula = "cum_lick_pct ~ time_min * condition * laser"

    # Fit model
    model = smf.mixedlm(formula, df_grp, groups=df_grp["mouse"])
    result = model.fit()
    print(result.summary())

    # Extract fixed effects table - it's already a DataFrame
    coefs_df = result.summary().tables[1].copy()

    # Clean up & add metadata
    coefs_df["group"] = grp
    
    # Reset index to make the parameter names a regular column
    coefs_df = coefs_df.reset_index()
    
    coefs_df.rename(columns={
        "index": "Parameter",
        "Coef.": "Estimate",
        "Std.Err.": "SE",
        "P>|z|": "p_value"
    }, inplace=True)

    results_list.append(coefs_df)

# -----------------------------
# Combine and export to Excel
# -----------------------------
results_df = pd.concat(results_list, ignore_index=True)

# Optional: re-order columns for clarity
results_df = results_df[["group", "Parameter", "Estimate", "SE", "z", "p_value", "[0.025", "0.975]"]]

# Save
output_path = "../lick_LMM_results.xlsx"
results_df.to_excel(output_path, index=False)
print(f"\n✅ Results successfully exported to: {output_path}")
