import os.path

import pandas as pd
from openpyxl import Workbook

data_dir = "/Users/atanas/Documents/workspace/data/analysis/opto/fat"

# Load your data from an Excel file
# If your data is in a sheet named "Sheet1":
df = pd.read_excel(os.path.join(data_dir, "opto-results.xlsx"), sheet_name="5C-raw")

groups = {
    "nac": df[(df['area'] == 'nac') & (df['opsin'] == 'chrimson')],
    "bla": df[(df['area'] == 'bla') & (df['opsin'] == 'chrimson')],
    "control": df[(df['opsin'] == 'control')]
}

# exclude = ["656_3", "201_3", '662_4']
# exclude = ["522_3", "653_2", "653_5", "662_2", "662_4",
#             "201_2", "201_3", "652_3", "656_3"]

# BLA = 5
# NAc = 4
# Control = 6

for group, df_group in groups.items():
    df_group = df_group[~(df_group["excluded"] == 1)]

    response_cols = ["#trials",	"#trials_stim",	"#trials_no_stim", "%correct_stim",	"%incorrect_stim",	"%omissions_stim",	"%premature_stim",	"%correct_no_stim",	"%incorrect_no_stim",	"%omissions_no_stim",	"%premature_no_stim",	"%correct",	"%incorrect",	"%omissions",	"%premature", "%correct_Remmelink_calc", "%omissions_Remmelink_calc", "%premature_Remmelink_calc"]

    # Reshape to long format
    df_long = df_group.melt(
        id_vars=["mouse", "condition"],
        value_vars=response_cols,
        var_name="response",
        value_name="value"
    )

    # Pivot wider: one column per condition × response
    df_wide = df_long.pivot_table(
        index="mouse",
        columns=["condition", "response"],
        values="value"
    )

    # Flatten MultiIndex columns to condition_response style
    df_wide.columns = [f"{cond}_{resp}" for cond, resp in df_wide.columns]

    # Reset index so mouse is a column
    df_wide = df_wide.reset_index()
    # Export
    df_wide.to_excel(f"5c/prism_ready_5c_column_{group}.xlsx", merge_cells=False)

# --- Export with two header rows ---
# with pd.ExcelWriter("prism_ready.xlsx", engine="openpyxl") as writer:
#     # write index+columns manually so multi-level headers stay intact
#     df_wide.to_excel(writer, sheet_name="Sheet1", merge_cells=False)
