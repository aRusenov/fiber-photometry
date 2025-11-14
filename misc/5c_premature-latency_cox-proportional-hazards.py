# =============================================
# Clustered Cox Regression + Survival Curves (cleaner version)
# =============================================

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.2)
# ------------------------------
# 1. Load your data
# ------------------------------
df = pd.read_excel('/Users/atanas/Documents/workspace/data/analysis/opto/fat/opto-results.xlsx', sheet_name='5C-premature-latencies-Remm')


df = df.rename(columns={'condition': 'session', 'stim': 'laser'})
df['laser'] = df['laser'].replace({0: 'OFF', 1: 'ON'}).astype('category')

# ------------------------------
# 4. Plot survival curves
# ------------------------------
# Define style: two colors, dashed for Laser ON
style_map = {
    ('mock', 'OFF'): {'color': 'tab:blue', 'linestyle': ':'},  # Mock OFF
    ('mock', 'ON'): {'color': 'tab:blue', 'linestyle': '-'},  # Mock ON
    ('stim', 'OFF'): {'color': 'tab:red', 'linestyle': ':'},  # Stim OFF
    ('stim', 'ON'): {'color': 'tab:red', 'linestyle': '-'}  # Stim ON
}

label_map = {
    ('mock', 'OFF'): 'Mock (Laser OFF)',
    ('mock', 'ON'): 'Mock (Laser ON)',
    ('stim', 'OFF'): 'Stim (Laser OFF)',
    ('stim', 'ON'): 'Stim (Laser ON)'
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

groups = ['bla', 'nac', 'control']
panel_letters = ['A', 'B', 'C']

no_mock_prem_excl = [] # ['202_1', '204_1']
exclude = ["656_3", "201_3", '653_5', "201_2", "662_4"] + no_mock_prem_excl
for idx, (ax, group, letter) in enumerate(zip(axes, groups, panel_letters)):
    group_df = df[df['group'] == group]
    group_df = group_df[~group_df['mouse'].isin(exclude)]
    print(f"\nUnique mice in {group} group:", group_df['mouse'].unique())

    # ------------------------------
    # 2. Convert categorical variables
    # ------------------------------
    group_df['session'] = group_df['session'].astype('category')
    group_df['laser'] = group_df['laser'].astype('category')

    # Optional combined factor
    group_df['cond_stim'] = group_df['session'].astype(str) + '_' + df['laser'].astype(str)

    # ------------------------------
    # 3. Cox regression (with clustering)
    # ------------------------------
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        group_df,
        duration_col='latency',
        event_col='event',
        cluster_col='mouse',
        formula='session + laser + session:laser'
    )

    print(f"=== Cox Regression Summary for {group} ===")
    cph.print_summary()

    for (cond, stim), style in style_map.items():
        sub_df = group_df[(group_df['session'] == cond) & (group_df['laser'] == stim)]

        if len(sub_df) > 0:  # safety check
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=sub_df['latency'],
                event_observed=sub_df['event']
            )
            # Invert to cumulative incidence
            failure = 1 - kmf.survival_function_

            # Also get CI bounds and invert them
            ci = kmf.confidence_interval_
            ci_lower = 1 - ci.iloc[:, 1]  # upper bound of survival -> lower bound of failure
            ci_upper = 1 - ci.iloc[:, 0]  # lower bound of survival -> upper bound of failure

            # Plot the curve
            ax.plot(
                failure.index,
                failure.values,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=2,
                label=label_map[(cond, stim)],
            )

            # Shaded CI
            ax.fill_between(
                failure.index,
                ci_lower,
                ci_upper,
                color=style['color'],
                alpha=0.1
            )


    # Titles + axis cleanup
    # ax.set_title(f"{group.upper()}", fontsize=12, weight="bold")
    ax.set_xticks([2, 3, 4, 5, 6, 7])
    ax.set_xlim([1, 7])
    ax.set_ylim([0, 0.4])
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    if idx == 0:  # Only leftmost plot shows y label
        ax.set_ylabel("Cumulative probability\nof premature response\n", fontsize=14)
    else:
        ax.set_ylabel("")
    ax.set_xlabel("Premature response latency (seconds)" if idx == 1 else "", fontsize=14)
    if ax.get_legend():
        ax.get_legend().remove()

    # Panel letters
    # ax.text(-0.1, 1.05, letter, transform=ax.transAxes,
    #         fontsize=12, fontweight="bold", va='top', ha='right')

# Shared legend
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=12, ncol=4)
plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # make space for shared legend
plt.savefig(f"cox-km-visuals.svg")
plt.show()

##
# Filter out excluded mice
combined_df = df[~df["mouse"].isin(exclude)].copy()

# Ensure categorical coding
combined_df["session"] = combined_df["session"].astype("category")
combined_df["laser"] = combined_df["laser"].astype("category")
combined_df["group"] = combined_df["group"].astype("category")

# Fit full model with interactions
cph_all = CoxPHFitter(penalizer=0.1)
cph_all.fit(
    combined_df,
    duration_col="latency",
    event_col="event",
    cluster_col="mouse",
    formula="session * laser * group"
)

print(f"=== Cox Regression Summary between groups ===")
cph_all.print_summary()

print(f"Concordance = {cph_all.concordance_index_:.2f}")

# =============================================
# 5. Export Cox model summaries to Excel (rounded)
# =============================================

output_path = "/Users/atanas/Documents/workspace/data/analysis/opto/fat/cox_summary_results.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    # --- Loop over individual groups ---
    for group in groups:
        group_df = df[(df["group"] == group) & (~df["mouse"].isin(exclude))].copy()
        group_df["session"] = group_df["session"].astype("category")
        group_df["laser"] = group_df["laser"].astype("category")

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(
            group_df,
            duration_col="latency",
            event_col="event",
            cluster_col="mouse",
            formula="session + laser + session:laser"
        )

        # Extract and round summary
        summary_df = cph.summary.reset_index().rename(columns={"index": "covariate"})
        summary_df = summary_df.round(3)  # round to 3 decimals
        summary_df.to_excel(writer, sheet_name=f"{group}_summary", index=False)

    # --- Combined model ---
    combined_summary_df = cph_all.summary.reset_index().rename(columns={"index": "covariate"})
    combined_summary_df = combined_summary_df.round(3)
    combined_summary_df.to_excel(writer, sheet_name="combined_summary", index=False)

print(f"\n✅ Cox regression summaries (rounded to 3 decimals) exported to:\n{output_path}")
