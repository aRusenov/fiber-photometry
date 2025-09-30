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

exclude = ["656_3", "201_3", '652_3', "201_2", "662_4"]

for group, df_group in groups.items():
    # Melt to long
    df_group = df_group[~df_group["mouse"].isin(exclude)]
    df_long = df_group.melt(
        id_vars=["mouse", "condition"],
        value_vars=["%premature_stim", "%premature_no_stim"],
        var_name="measure",
        value_name="value"
    )

    # Pivot: rows = condition, columns = (measure, mouse)
    df_wide = df_long.pivot_table(
        index="condition",
        columns=["measure", "mouse"],
        values="value"
    )

    # Ensure nice column order
    df_wide = df_wide.sort_index(axis=1, level=[0, 1])
    # Export to Excel with multi-level headers
    df_wide.to_excel(f"prism_ready_5c_{group}.xlsx", merge_cells=False)

# --- Export with two header rows ---
# with pd.ExcelWriter("prism_ready.xlsx", engine="openpyxl") as writer:
#     # write index+columns manually so multi-level headers stay intact
#     df_wide.to_excel(writer, sheet_name="Sheet1", merge_cells=False)
