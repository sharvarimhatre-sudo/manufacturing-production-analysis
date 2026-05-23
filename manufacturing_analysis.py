"""
Manufacturing Production Data Analysis
=======================================
Dataset: https://www.kaggle.com/datasets/ziya07/manufacturing-production-data
KPIs: Schedule Adherence, Capacity Utilization, Lead Time, Resource Utilization,
      Efficiency Score Distribution
Visualizations: Production Schedule vs Actual, Resource Allocation Heatmap,
                Efficiency Segmentation, Delay Root Causes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  SECTION 0 — DATA GENERATION (mirrors Kaggle dataset schema)
#  Replace `df = generate_dataset()` with:
#  df = pd.read_csv("manufacturing_production_data.csv")
# ─────────────────────────────────────────────

np.random.seed(42)

def generate_dataset(n=500):
    """
    Generates a synthetic dataset matching the Kaggle Manufacturing Production Data schema.
    Columns match: Job_ID, Machine_ID, Department, Shift, Scheduled_Start,
    Scheduled_End, Actual_Start, Actual_End, Scheduled_Qty, Actual_Qty,
    Machine_Capacity, Workers_Assigned, Max_Workers, Efficiency_Score, Delay_Reason
    """
    departments  = ["Assembly", "Machining", "Fabrication", "Quality", "Packaging"]
    shifts       = ["Morning", "Afternoon", "Night"]
    machines     = [f"M{str(i).zfill(3)}" for i in range(1, 21)]
    delay_cats   = [
        "Material Shortage", "Machine Breakdown", "Labour Absence",
        "Power Outage", "Quality Rework", "No Delay", "No Delay",
        "No Delay", "No Delay", "No Delay"           # weight toward No Delay
    ]

    base_date = pd.Timestamp("2024-01-01")
    sched_starts = [base_date + pd.Timedelta(hours=np.random.randint(0, 24*180))
                    for _ in range(n)]
    sched_durations = np.random.randint(2, 12, n)          # hours
    sched_ends   = [s + pd.Timedelta(hours=int(d))
                    for s, d in zip(sched_starts, sched_durations)]

    delay_hours  = np.where(np.random.rand(n) < 0.35,
                            np.random.exponential(3, n), 0).round(1)

    actual_starts = [s + pd.Timedelta(hours=float(dh))
                     for s, dh in zip(sched_starts, delay_hours)]
    actual_durations = sched_durations * np.random.uniform(0.85, 1.40, n)
    actual_ends   = [s + pd.Timedelta(hours=float(d))
                     for s, d in zip(actual_starts, actual_durations)]

    machine_cap   = np.random.randint(80, 200, n)
    actual_qty    = (machine_cap * np.random.uniform(0.55, 1.05, n)).astype(int)
    scheduled_qty = (machine_cap * np.random.uniform(0.85, 1.00, n)).astype(int)
    actual_qty    = np.minimum(actual_qty, machine_cap)

    max_workers  = np.random.randint(5, 20, n)
    workers_assigned = (max_workers * np.random.uniform(0.5, 1.0, n)).astype(int)

    efficiency   = np.clip(
        np.random.normal(loc=72, scale=15, size=n), 20, 100
    ).round(1)

    delay_reason = np.where(
        delay_hours > 0,
        np.random.choice(delay_cats[:5], n),
        "No Delay"
    )

    df = pd.DataFrame({
        "Job_ID"          : [f"JOB-{i:04d}" for i in range(1, n+1)],
        "Machine_ID"      : np.random.choice(machines, n),
        "Department"      : np.random.choice(departments, n),
        "Shift"           : np.random.choice(shifts, n),
        "Scheduled_Start" : sched_starts,
        "Scheduled_End"   : sched_ends,
        "Actual_Start"    : actual_starts,
        "Actual_End"      : actual_ends,
        "Scheduled_Qty"   : scheduled_qty,
        "Actual_Qty"      : actual_qty,
        "Machine_Capacity": machine_cap,
        "Workers_Assigned": workers_assigned,
        "Max_Workers"     : max_workers,
        "Efficiency_Score": efficiency,
        "Delay_Hours"     : delay_hours,
        "Delay_Reason"    : delay_reason,
    })
    return df


# ── Load data ──────────────────────────────────────────────────────────────────
df = generate_dataset(500)

# ── Derived columns ────────────────────────────────────────────────────────────
df["Sched_Duration_hrs"] = (df["Scheduled_End"] - df["Scheduled_Start"]).dt.total_seconds() / 3600
df["Actual_Duration_hrs"] = (df["Actual_End"] - df["Actual_Start"]).dt.total_seconds() / 3600
df["Lead_Time_hrs"]       = (df["Actual_End"] - df["Scheduled_Start"]).dt.total_seconds() / 3600
df["Sched_Lead_Time_hrs"] = df["Sched_Duration_hrs"]
df["On_Time"]             = df["Actual_End"] <= df["Scheduled_End"]
df["Capacity_Util_pct"]   = (df["Actual_Qty"] / df["Machine_Capacity"] * 100).round(2)
df["Resource_Util_pct"]   = (df["Workers_Assigned"] / df["Max_Workers"] * 100).round(2)
df["Month"]               = df["Scheduled_Start"].dt.to_period("M").astype(str)
df["Week"]                = df["Scheduled_Start"].dt.isocalendar().week.astype(int)

# ─────────────────────────────────────────────
#  SECTION 1 — KPI CALCULATIONS
# ─────────────────────────────────────────────

print("=" * 70)
print("  MANUFACTURING PRODUCTION — KPI DASHBOARD")
print("=" * 70)

# 1. Schedule Adherence
on_time_rate       = df["On_Time"].mean() * 100
avg_delay          = df.loc[~df["On_Time"], "Delay_Hours"].mean()
dept_adherence     = df.groupby("Department")["On_Time"].mean().mul(100).round(1)

print(f"\n📌 KPI-1  SCHEDULE ADHERENCE")
print(f"   Overall On-Time Rate     : {on_time_rate:.1f}%")
print(f"   Avg Delay (late jobs)    : {avg_delay:.2f} hrs")
print(f"   By Department:")
for dept, val in dept_adherence.items():
    bar = "█" * int(val / 5)
    print(f"     {dept:<15} {val:>5.1f}%  {bar}")

# 2. Capacity Utilization
avg_cap_util   = df["Capacity_Util_pct"].mean()
dept_cap_util  = df.groupby("Department")["Capacity_Util_pct"].mean().round(1)

print(f"\n📌 KPI-2  CAPACITY UTILIZATION")
print(f"   Overall Avg Utilization  : {avg_cap_util:.1f}%")
print(f"   By Department:")
for dept, val in dept_cap_util.items():
    bar = "█" * int(val / 5)
    print(f"     {dept:<15} {val:>5.1f}%  {bar}")

# 3. Lead Time
avg_lead_time  = df["Lead_Time_hrs"].mean()
p50_lt         = df["Lead_Time_hrs"].median()
p95_lt         = df["Lead_Time_hrs"].quantile(0.95)
dept_lead_time = df.groupby("Department")["Lead_Time_hrs"].mean().round(2)

print(f"\n📌 KPI-3  LEAD TIME (hrs)")
print(f"   Mean Lead Time           : {avg_lead_time:.2f} hrs")
print(f"   Median (P50)             : {p50_lt:.2f} hrs")
print(f"   95th Percentile (P95)    : {p95_lt:.2f} hrs")
print(f"   By Department:")
for dept, val in dept_lead_time.items():
    print(f"     {dept:<15} {val:>6.2f} hrs")

# 4. Resource Utilization
avg_res_util  = df["Resource_Util_pct"].mean()
shift_res_util = df.groupby("Shift")["Resource_Util_pct"].mean().round(1)
dept_res_util  = df.groupby("Department")["Resource_Util_pct"].mean().round(1)

print(f"\n📌 KPI-4  RESOURCE UTILIZATION")
print(f"   Overall Avg Utilization  : {avg_res_util:.1f}%")
print(f"   By Shift:")
for shift, val in shift_res_util.items():
    bar = "█" * int(val / 5)
    print(f"     {shift:<12} {val:>5.1f}%  {bar}")

# 5. Efficiency Score Distribution
eff_mean   = df["Efficiency_Score"].mean()
eff_std    = df["Efficiency_Score"].std()
eff_median = df["Efficiency_Score"].median()
# Segments
df["Eff_Segment"] = pd.cut(
    df["Efficiency_Score"],
    bins=[0, 50, 65, 80, 100],
    labels=["Critical (<50)", "Low (50–65)", "Moderate (65–80)", "High (>80)"]
)
seg_counts = df["Eff_Segment"].value_counts().sort_index()

print(f"\n📌 KPI-5  EFFICIENCY SCORE DISTRIBUTION")
print(f"   Mean                     : {eff_mean:.1f}")
print(f"   Std Dev                  : {eff_std:.1f}")
print(f"   Median                   : {eff_median:.1f}")
print(f"   Segment Breakdown:")
for seg, cnt in seg_counts.items():
    pct = cnt / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"     {str(seg):<22} {cnt:>4} jobs  ({pct:5.1f}%)  {bar}")

print("\n" + "=" * 70)

# ─────────────────────────────────────────────
#  SECTION 2 — VISUALIZATIONS
# ─────────────────────────────────────────────

# ── Styling ────────────────────────────────────────────────────────────────────
DARK_BG  = "#0F1117"
PANEL_BG = "#161B22"
ACCENT1  = "#00D4FF"     # cyan
ACCENT2  = "#FF6B35"     # orange
ACCENT3  = "#7FBA00"     # green
ACCENT4  = "#FFB800"     # amber
TEXT_CLR = "#E6EDF3"
GRID_CLR = "#21262D"

plt.rcParams.update({
    "figure.facecolor" : DARK_BG,
    "axes.facecolor"   : PANEL_BG,
    "axes.edgecolor"   : GRID_CLR,
    "axes.labelcolor"  : TEXT_CLR,
    "axes.titlecolor"  : TEXT_CLR,
    "xtick.color"      : TEXT_CLR,
    "ytick.color"      : TEXT_CLR,
    "text.color"       : TEXT_CLR,
    "grid.color"       : GRID_CLR,
    "grid.alpha"       : 0.5,
    "font.family"      : "monospace",
})

# ══════════════════════════════════════════════════════════════════════════════
#  VIZ-1  PRODUCTION SCHEDULE vs ACTUAL  (4-panel overview)
# ══════════════════════════════════════════════════════════════════════════════

fig1, axes = plt.subplots(2, 2, figsize=(18, 10))
fig1.patch.set_facecolor(DARK_BG)
fig1.suptitle("PRODUCTION  ·  SCHEDULE vs ACTUAL", fontsize=18,
              fontweight="bold", color=TEXT_CLR, y=0.98)

# Panel A — Monthly Scheduled vs Actual Qty
ax = axes[0, 0]
monthly = df.groupby("Month")[["Scheduled_Qty", "Actual_Qty"]].sum().reset_index()
x       = np.arange(len(monthly))
w       = 0.35
bars1 = ax.bar(x - w/2, monthly["Scheduled_Qty"], w, color=ACCENT1, alpha=0.85, label="Scheduled")
bars2 = ax.bar(x + w/2, monthly["Actual_Qty"],    w, color=ACCENT2, alpha=0.85, label="Actual")
ax.set_xticks(x)
ax.set_xticklabels(monthly["Month"], rotation=45, ha="right", fontsize=7)
ax.set_title("Monthly Qty: Scheduled vs Actual", fontsize=11, pad=10)
ax.set_ylabel("Total Units")
ax.legend(framealpha=0.2)
ax.yaxis.grid(True, linestyle="--")
ax.set_axisbelow(True)

# Panel B — Duration Scatter: Scheduled vs Actual
ax = axes[0, 1]
colors_scatter = np.where(df["On_Time"], ACCENT3, ACCENT2)
sc = ax.scatter(df["Sched_Duration_hrs"], df["Actual_Duration_hrs"],
                c=colors_scatter, alpha=0.4, s=18, edgecolors="none")
lim = max(df["Sched_Duration_hrs"].max(), df["Actual_Duration_hrs"].max()) + 1
ax.plot([0, lim], [0, lim], color=ACCENT1, lw=1.5, linestyle="--", label="Perfect Schedule")
ax.set_xlabel("Scheduled Duration (hrs)")
ax.set_ylabel("Actual Duration (hrs)")
ax.set_title("Job Duration: Scheduled vs Actual", fontsize=11, pad=10)
patch_ot  = mpatches.Patch(color=ACCENT3, label="On-Time")
patch_late= mpatches.Patch(color=ACCENT2, label="Late")
ax.legend(handles=[patch_ot, patch_late, 
          plt.Line2D([0],[0], color=ACCENT1, lw=1.5, linestyle="--", label="Perfect")],
          framealpha=0.2, fontsize=8)
ax.grid(True, linestyle="--")

# Panel C — On-Time Rate by Department
ax = axes[1, 0]
dept_order  = dept_adherence.sort_values().index.tolist()
vals        = dept_adherence[dept_order].values
bar_colors  = [ACCENT3 if v >= 70 else ACCENT4 if v >= 55 else ACCENT2 for v in vals]
bars = ax.barh(dept_order, vals, color=bar_colors, edgecolor=DARK_BG, linewidth=0.5)
ax.axvline(70, color=ACCENT1, linestyle="--", lw=1.2, label="Target 70%")
for bar, val in zip(bars, vals):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9)
ax.set_xlim(0, 105)
ax.set_title("Schedule Adherence by Department (%)", fontsize=11, pad=10)
ax.set_xlabel("On-Time Rate (%)")
ax.legend(framealpha=0.2)
ax.xaxis.grid(True, linestyle="--")
ax.set_axisbelow(True)

# Panel D — Delay Distribution (histogram)
ax = axes[1, 1]
delayed = df.loc[df["Delay_Hours"] > 0, "Delay_Hours"]
ax.hist(delayed, bins=25, color=ACCENT2, alpha=0.8, edgecolor=DARK_BG)
ax.axvline(delayed.mean(), color=ACCENT1, linestyle="--", lw=1.5,
           label=f"Mean {delayed.mean():.1f}h")
ax.axvline(delayed.median(), color=ACCENT4, linestyle="--", lw=1.5,
           label=f"Median {delayed.median():.1f}h")
ax.set_title("Delay Distribution (Late Jobs Only)", fontsize=11, pad=10)
ax.set_xlabel("Delay (hrs)")
ax.set_ylabel("Job Count")
ax.legend(framealpha=0.2)
ax.yaxis.grid(True, linestyle="--")
ax.set_axisbelow(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/mnt/user-data/outputs/viz1_schedule_vs_actual.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("✅  viz1_schedule_vs_actual.png saved")


# ══════════════════════════════════════════════════════════════════════════════
#  VIZ-2  RESOURCE ALLOCATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

fig2, axes = plt.subplots(1, 2, figsize=(18, 7))
fig2.patch.set_facecolor(DARK_BG)
fig2.suptitle("RESOURCE  ALLOCATION  HEATMAP", fontsize=18,
              fontweight="bold", color=TEXT_CLR, y=1.01)

# Panel A — Dept × Shift: avg resource utilisation
ax = axes[0]
pivot_res = df.pivot_table(
    values="Resource_Util_pct", index="Department",
    columns="Shift", aggfunc="mean"
)
cmap_res = LinearSegmentedColormap.from_list(
    "res", ["#0F1117", "#00558A", ACCENT1, ACCENT3], N=256
)
sns.heatmap(pivot_res, ax=ax, cmap=cmap_res, annot=True, fmt=".1f",
            linewidths=0.4, linecolor=DARK_BG,
            cbar_kws={"label": "Resource Util %", "shrink": 0.8},
            annot_kws={"size": 11, "weight": "bold"})
ax.set_title("Resource Utilisation  (Dept × Shift)", fontsize=12, pad=12)
ax.set_xlabel("Shift", labelpad=8)
ax.set_ylabel("Department", labelpad=8)
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)

# Panel B — Machine-level capacity utilisation
ax = axes[1]
pivot_mach = df.pivot_table(
    values="Capacity_Util_pct", index="Machine_ID",
    columns="Department", aggfunc="mean"
).fillna(0)
cmap_cap = LinearSegmentedColormap.from_list(
    "cap", ["#0F1117", "#6B0F1A", ACCENT2, ACCENT4], N=256
)
sns.heatmap(pivot_mach, ax=ax, cmap=cmap_cap, annot=False,
            linewidths=0.2, linecolor=DARK_BG,
            cbar_kws={"label": "Capacity Util %", "shrink": 0.8})
ax.set_title("Machine Capacity Utilisation  (Machine × Dept)", fontsize=12, pad=12)
ax.set_xlabel("Department", labelpad=8)
ax.set_ylabel("Machine ID", labelpad=8)
ax.tick_params(axis="x", rotation=20)
ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/viz2_resource_heatmap.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("✅  viz2_resource_heatmap.png saved")


# ══════════════════════════════════════════════════════════════════════════════
#  VIZ-3  EFFICIENCY SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

fig3 = plt.figure(figsize=(18, 10))
fig3.patch.set_facecolor(DARK_BG)
fig3.suptitle("EFFICIENCY  SCORE  SEGMENTATION", fontsize=18,
              fontweight="bold", color=TEXT_CLR, y=0.99)

gs = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.40, wspace=0.35)

SEG_COLORS = {
    "Critical (<50)" : "#FF2D55",
    "Low (50–65)"    : ACCENT2,
    "Moderate (65–80)": ACCENT4,
    "High (>80)"     : ACCENT3,
}

# Panel A — KDE distribution
ax_kde = fig3.add_subplot(gs[0, :2])
for seg, color in SEG_COLORS.items():
    subset = df.loc[df["Eff_Segment"] == seg, "Efficiency_Score"]
    if len(subset) > 1:
        kde_x = np.linspace(subset.min(), subset.max(), 200)
        kde   = stats.gaussian_kde(subset)(kde_x)
        ax_kde.fill_between(kde_x, kde, alpha=0.35, color=color)
        ax_kde.plot(kde_x, kde, color=color, lw=1.5, label=seg)
for cut in [50, 65, 80]:
    ax_kde.axvline(cut, color=TEXT_CLR, lw=0.8, linestyle=":")
ax_kde.set_title("Efficiency Score Distribution (KDE)", fontsize=12, pad=10)
ax_kde.set_xlabel("Efficiency Score")
ax_kde.set_ylabel("Density")
ax_kde.legend(framealpha=0.15, fontsize=9)
ax_kde.yaxis.grid(True, linestyle="--")
ax_kde.set_axisbelow(True)

# Panel B — Donut chart (segment share)
ax_donut = fig3.add_subplot(gs[0, 2])
wedge_sizes  = seg_counts.values
wedge_labels = [f"{s}\n({c})" for s, c in zip(seg_counts.index, wedge_sizes)]
wedge_colors = [SEG_COLORS[s] for s in seg_counts.index]
wedges, texts, autotexts = ax_donut.pie(
    wedge_sizes, labels=wedge_labels, colors=wedge_colors,
    autopct="%1.1f%%", pctdistance=0.78, startangle=140,
    wedgeprops={"width": 0.5, "edgecolor": DARK_BG, "linewidth": 1.5}
)
for t  in texts    : t.set_fontsize(8);  t.set_color(TEXT_CLR)
for at in autotexts: at.set_fontsize(8); at.set_color(DARK_BG); at.set_fontweight("bold")
ax_donut.set_title("Segment Share", fontsize=12, pad=10)

# Panel C — Efficiency by Department (box)
ax_box = fig3.add_subplot(gs[1, :2])
departments   = df["Department"].unique().tolist()
dept_eff_data = [df.loc[df["Department"] == d, "Efficiency_Score"].values
                 for d in departments]
bp = ax_box.boxplot(dept_eff_data, labels=departments, patch_artist=True,
                    medianprops={"color": DARK_BG, "linewidth": 2},
                    whiskerprops={"color": TEXT_CLR},
                    capprops={"color": TEXT_CLR},
                    flierprops={"marker": "o", "markerfacecolor": ACCENT2,
                                "markersize": 4, "alpha": 0.5})
box_colors = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, "#9B59B6"]
for patch, col in zip(bp["boxes"], box_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)
ax_box.axhline(eff_mean, color=TEXT_CLR, linestyle="--", lw=1,
               label=f"Overall Mean {eff_mean:.1f}")
ax_box.set_title("Efficiency Score by Department", fontsize=12, pad=10)
ax_box.set_ylabel("Efficiency Score")
ax_box.legend(framealpha=0.15, fontsize=9)
ax_box.yaxis.grid(True, linestyle="--")
ax_box.set_axisbelow(True)

# Panel D — Shift efficiency
ax_shift = fig3.add_subplot(gs[1, 2])
shift_eff = df.groupby("Shift")["Efficiency_Score"].mean().sort_values(ascending=True)
shift_colors = [ACCENT3 if v >= eff_mean else ACCENT2 for v in shift_eff.values]
ax_shift.barh(shift_eff.index, shift_eff.values, color=shift_colors, edgecolor=DARK_BG)
ax_shift.axvline(eff_mean, color=ACCENT1, linestyle="--", lw=1.2,
                 label=f"Avg {eff_mean:.1f}")
for i, (idx, val) in enumerate(shift_eff.items()):
    ax_shift.text(val + 0.3, i, f"{val:.1f}", va="center", fontsize=10)
ax_shift.set_xlim(0, 100)
ax_shift.set_title("Avg Efficiency by Shift", fontsize=12, pad=10)
ax_shift.set_xlabel("Efficiency Score")
ax_shift.legend(framealpha=0.15, fontsize=9)
ax_shift.xaxis.grid(True, linestyle="--")
ax_shift.set_axisbelow(True)

plt.savefig("/mnt/user-data/outputs/viz3_efficiency_segmentation.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("✅  viz3_efficiency_segmentation.png saved")


# ══════════════════════════════════════════════════════════════════════════════
#  VIZ-4  DELAY ROOT CAUSES
# ══════════════════════════════════════════════════════════════════════════════

delayed_df  = df[df["Delay_Reason"] != "No Delay"].copy()
cause_stats = delayed_df.groupby("Delay_Reason").agg(
    Count=("Job_ID", "count"),
    Avg_Delay=("Delay_Hours", "mean"),
    Total_Delay=("Delay_Hours", "sum"),
).sort_values("Count", ascending=False)

fig4, axes = plt.subplots(2, 2, figsize=(18, 10))
fig4.patch.set_facecolor(DARK_BG)
fig4.suptitle("DELAY  ROOT  CAUSE  ANALYSIS", fontsize=18,
              fontweight="bold", color=TEXT_CLR, y=0.99)

CAUSE_COLORS = {
    "Material Shortage" : ACCENT2,
    "Machine Breakdown" : "#FF2D55",
    "Labour Absence"    : ACCENT4,
    "Power Outage"      : ACCENT1,
    "Quality Rework"    : "#9B59B6",
}
palette = [CAUSE_COLORS.get(c, TEXT_CLR) for c in cause_stats.index]

# Panel A — Pareto (count + cumulative %)
ax = axes[0, 0]
cumulative = cause_stats["Count"].cumsum() / cause_stats["Count"].sum() * 100
ax.bar(cause_stats.index, cause_stats["Count"], color=palette, edgecolor=DARK_BG, alpha=0.9)
ax2 = ax.twinx()
ax2.plot(cause_stats.index, cumulative, color=ACCENT1, marker="D",
         markersize=6, lw=2, label="Cumulative %")
ax2.axhline(80, color=TEXT_CLR, lw=0.8, linestyle=":", alpha=0.6, label="80% Line")
ax2.set_ylim(0, 110)
ax2.set_ylabel("Cumulative %", color=TEXT_CLR)
ax2.tick_params(axis="y", labelcolor=TEXT_CLR)
ax2.legend(loc="lower right", framealpha=0.15, fontsize=8)
ax.set_title("Pareto: Delay Causes by Job Count", fontsize=12, pad=10)
ax.set_ylabel("Job Count")
ax.tick_params(axis="x", rotation=15)
ax.yaxis.grid(True, linestyle="--")
ax.set_axisbelow(True)

# Panel B — Avg delay hours by cause
ax = axes[0, 1]
bars = ax.bar(cause_stats.index, cause_stats["Avg_Delay"],
              color=palette, edgecolor=DARK_BG, alpha=0.9)
for bar, val in zip(bars, cause_stats["Avg_Delay"]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
            f"{val:.1f}h", ha="center", fontsize=9)
ax.set_title("Average Delay Duration by Cause (hrs)", fontsize=12, pad=10)
ax.set_ylabel("Avg Delay (hrs)")
ax.tick_params(axis="x", rotation=15)
ax.yaxis.grid(True, linestyle="--")
ax.set_axisbelow(True)

# Panel C — Delay causes by Department (stacked 100%)
ax = axes[1, 0]
dept_cause = (
    delayed_df.groupby(["Department", "Delay_Reason"])
    .size().unstack(fill_value=0)
)
dept_cause_pct = dept_cause.div(dept_cause.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(dept_cause_pct))
for cause in dept_cause_pct.columns:
    col = CAUSE_COLORS.get(cause, TEXT_CLR)
    ax.bar(dept_cause_pct.index, dept_cause_pct[cause],
           bottom=bottom, color=col, edgecolor=DARK_BG, linewidth=0.4,
           label=cause, alpha=0.9)
    bottom += dept_cause_pct[cause].values
ax.set_title("Delay Root Causes by Department (%)", fontsize=12, pad=10)
ax.set_ylabel("Share (%)")
ax.set_ylim(0, 105)
ax.legend(loc="lower right", framealpha=0.15, fontsize=8, ncol=1)
ax.tick_params(axis="x", rotation=15)
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

# Panel D — Delay impact on efficiency
ax = axes[1, 1]
for cause, color in CAUSE_COLORS.items():
    subset = delayed_df[delayed_df["Delay_Reason"] == cause]
    if len(subset):
        ax.scatter(subset["Delay_Hours"], subset["Efficiency_Score"],
                   color=color, alpha=0.55, s=22, label=cause, edgecolors="none")
# regression line
m, b, r, p, _ = stats.linregress(delayed_df["Delay_Hours"],
                                   delayed_df["Efficiency_Score"])
x_fit = np.linspace(0, delayed_df["Delay_Hours"].max(), 100)
ax.plot(x_fit, m * x_fit + b, color=TEXT_CLR, lw=1.5, linestyle="--",
        label=f"Trend (r={r:.2f})")
ax.set_title("Delay Hours vs Efficiency Score", fontsize=12, pad=10)
ax.set_xlabel("Delay (hrs)")
ax.set_ylabel("Efficiency Score")
ax.legend(framealpha=0.15, fontsize=8, markerscale=1.5)
ax.grid(True, linestyle="--")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("/mnt/user-data/outputs/viz4_delay_root_causes.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("✅  viz4_delay_root_causes.png saved")

print("\n✅  All 4 visualisations saved to /mnt/user-data/outputs/")
