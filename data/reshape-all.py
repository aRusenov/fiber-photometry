import os

import pandas as pd

data_dir = "/Users/atanas/Documents/workspace/data/analysis/opto/fat"

# Load your data from an Excel file
# If your data is in a sheet named "Sheet1":
df = pd.read_excel(os.path.join(data_dir, "opto-results.xlsx"), sheet_name="5C-raw")

df = df[~(df["excluded"] == 1)]
# --- 1. Clean column names: replace '#' with 'nr_' ---
df = df.rename(columns=lambda x: x.replace("#", "nr_"))

# --- 2. Determine stim_week for each mouse ---
# Convert date columns to datetime
df["date_start"] = pd.to_datetime(df["date_start"])

# For each mouse, check if stim date_start is earlier than mock date_start
def assign_stim_week(subdf):
    stim_date = subdf.loc[subdf["condition"] == "stim", "date_start"].min()
    mock_date = subdf.loc[subdf["condition"] == "mock", "date_start"].min()
    stim_week = 0 if stim_date < mock_date else 1
    subdf = subdf.assign(stim_week=stim_week)
    return subdf

df = df.groupby("mouse", group_keys=False).apply(assign_stim_week)

# --- 3. Define ID vars (excluding date_start, date_end now) ---
id_vars = ["mouse", "area", "opsin", "box", "stim_week", "group", "batch"]

# All other columns (metrics)
value_vars = [c for c in df.columns if c not in id_vars + ["condition", "date_start"]]

# --- 4. Reshape to long ---
df_long = df.melt(
    id_vars=id_vars + ["condition"],
    value_vars=value_vars,
    var_name="metric",
    value_name="value"
)

# --- 5. Pivot wider: rows = identifiers, columns = condition × metric ---
df_wide = df_long.pivot_table(
    index=id_vars,
    columns=["condition", "metric"],
    values="value"
)

# --- 6. Flatten MultiIndex columns ---
df_wide.columns = [f"{cond}_{metric}" for cond, metric in df_wide.columns]

# --- 7. Reset index ---
df_wide = df_wide.reset_index()

# Save
df_wide.to_excel(f"jasp_5c.xlsx", merge_cells=False)

