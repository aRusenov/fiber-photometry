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
    cols = list(df_group.filter(like="bin").columns)

    # Create long format dataframe
    df_long = pd.melt(df_group,
                      id_vars=['mouse', 'group', 'condition'],
                      value_vars=cols,
                      var_name='bin',
                      value_name='bin_value')

    # Extract numeric values from bin column and divide by 180
    df_long['bin'] = df_long['bin'].str.extract('(\d+)').astype(float) / 180
    df_long['stim'] = df_long['bin'].apply(lambda x: 0 if x % 2 == 0 else 1)

    # Export to Excel
    df_long.to_excel(f"fat/prism_ready_fat_binned_{group}.xlsx", index=False)

# --- Export with two header rows ---
# with pd.ExcelWriter("prism_ready.xlsx", engine="openpyxl") as writer:
#     # write index+columns manually so multi-level headers stay intact
#     df_wide.to_excel(writer, sheet_name="Sheet1", merge_cells=False)
