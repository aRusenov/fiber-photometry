import os.path

import pandas as pd
from openpyxl import Workbook

data_dir = "/Users/atanas/Documents/workspace/data/analysis/opto/fat"

# Load your data from an Excel file
# If your data is in a sheet named "Sheet1":
df = pd.read_excel(os.path.join(data_dir, "opto-results.xlsx"), sheet_name="fat-raw")

groups = {
    "nac": df[(df['area'] == 'nac') & (df['opsin'] == 'chrimson')],
    "bla": df[(df['area'] == 'bla') & (df['opsin'] == 'chrimson')],
    "control": df[(df['opsin'] == 'control')]
}


# exclude = ["522_3", "653_2", "653_5", "662_2", "662_4",
#             "201_2", "201_3", "652_3", "656_3"]

for group, df_group in groups.items():
    # df_group = df_group[~df_group["mouse"].isin(exclude)]
    df_group = df_group[~(df_group["excluded"] == 1)]
    melt_cols = list(df_group.filter(like="bin").columns) + ["total_licks", "ON", "OFF"]

    # Melt into long format
    df_long = df_group.melt(
        id_vars=[c for c in df_group.columns if c not in melt_cols],
        value_vars=melt_cols,
        var_name="response",
        value_name="value"
    )

    # Pivot wider: one column per condition × response
    df_wide = df_long.pivot_table(
        index=["mouse", "group"],
        columns=["condition", "response"],
        values="value"
    )
    

    # Flatten MultiIndex columns to condition_response style
    df_wide.columns = [f"{cond}_{resp}" for cond, resp in df_wide.columns]

    # Reset index so mouse and group are columns
    df_wide = df_wide.reset_index()

    # Export to Excel with multi-level headers
    df_wide.to_excel(f"fat/prism_ready_fat_binned_{group}.xlsx", merge_cells=False)

# --- Export with two header rows ---
# with pd.ExcelWriter("prism_ready.xlsx", engine="openpyxl") as writer:
#     # write index+columns manually so multi-level headers stay intact
#     df_wide.to_excel(writer, sheet_name="Sheet1", merge_cells=False)
