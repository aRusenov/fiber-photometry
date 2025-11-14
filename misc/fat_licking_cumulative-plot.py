import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# Load dataset
# =========================================
df = pd.read_csv("../data/fat-licks_long-cumulative.csv")
exclude = ["656_3", "201_3", '653_5', "201_2", "662_4"]
df = df[~df['mouse'].isin(exclude)]
# TODO: change group based on desired plot (bla | nac | control)
df = df[df["group"] == 'control']

print("Preview of data:")
print(df.head())
print(df.dtypes)

# Ensure correct dtypes
df["condition"] = df["condition"].astype(str)
df["laser"] = df["laser"].astype(str)
df["time_min"] = pd.to_numeric(df["time_min"], errors="coerce")
df["cum_lick_pct"] = pd.to_numeric(df["cum_lick_pct"], errors="coerce")

print("\nSummary stats for cum_lick_pct:")
print(df["cum_lick_pct"].describe())

# =========================================
# Summarize across mice
# =========================================
summary = (
    df.groupby(["condition", "time_min"])
    .agg(
        mean_pct=("cum_lick_pct", "mean"),
        sem_pct=("cum_lick_pct", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    )
    .reset_index()
)

print("\nSummary table preview:")
print(summary.head())

# =========================================
# Plot cumulative curves
# =========================================
plt.figure(figsize=(4, 4))
sns.set(style="whitegrid", font_scale=1.2)

palette = {
    ("Mock"): "blue",
    ("Stim"): "red",
}

for (cond), subset in summary.groupby(["condition"]):
    if subset.empty:
        print(f"⚠️ No data for {cond}, skipping.")
        continue

    label = f"{cond[0].capitalize()}"
    color = palette.get((cond), None)

    plt.plot(subset["time_min"], subset["mean_pct"], label=label, color=color, linewidth=2)
    plt.fill_between(
        subset["time_min"],
        subset["mean_pct"] - subset["sem_pct"],
        subset["mean_pct"] + subset["sem_pct"],
        alpha=0.2,
        color=color
    )

# plt.title("Cumulative Licking Curves by Session × Laser")
plt.xlabel("Time (min)", fontsize=12, weight='bold')
plt.ylabel("Cumulative Licks (% of total)", fontsize=12, weight='bold')
plt.ylim(0, 101)
plt.xlim(0, 60)
plt.xticks(range(0, 61, 10)[1:])
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
