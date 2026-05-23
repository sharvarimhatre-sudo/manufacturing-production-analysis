# 🏭 Manufacturing Production Analysis
### KPI Engineering + Operational Visualizations in Python

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-4c72b0?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Overview

A end-to-end Python analytics pipeline built on the
[Manufacturing Production Dataset](https://www.kaggle.com/datasets/ziya07/manufacturing-production-data)
from Kaggle. The project engineers **5 operational KPIs** from raw job-level
production data and produces **4 publication-quality visualizations** — all without
a BI tool.

The analysis covers **500 jobs** across **5 departments**, **20 machines**, and
**3 shifts**, exposing the hidden mechanics of a production floor where 82% of
jobs miss their scheduled completion time.

---

## 📊 Key Findings

| KPI | Result | Status |
|-----|--------|--------|
| Schedule Adherence | **18.0%** on-time rate | 🔴 Critical |
| Capacity Utilization | **79.5%** overall avg | 🟡 Misleading aggregate |
| Lead Time P95 | **15.19 hrs** vs 8.25 hrs mean | 🟡 High tail risk |
| Resource Utilization | **68.9%** — flat across all shifts | 🟠 Sequencing problem |
| Efficiency Score (mean) | **71.7 / 100** — 33.8% in low/critical band | 🔵 Systems issue |

> **Top insight:** 2 root causes — Material Shortage + Machine Breakdown —
> drive ~80% of all delay incidents. Pareto holds.

---

## 🗂️ Project Structure

```
manufacturing-production-analysis/
│
├── manufacturing_analysis.py       # Main pipeline — KPIs + all visualizations
│
├── outputs/
│   ├── viz1_schedule_vs_actual.png     # Monthly qty bars, duration scatter,
│   │                                   # adherence by dept, delay histogram
│   ├── viz2_resource_heatmap.png       # Dept×Shift + Machine×Dept heatmaps
│   ├── viz3_efficiency_segmentation.png# KDE curves, donut, box plots, shift bars
│   └── viz4_delay_root_causes.png      # Pareto, avg delay, dept stack, scatter
│
├── manufacturing_linkedin_article.md   # 2,500-word article + insight summary
└── README.md
```

---

## ⚙️ KPIs Engineered

### 1 — Schedule Adherence
Measures whether jobs complete by their scheduled end time.
```
On-Time Rate     = jobs where Actual_End ≤ Scheduled_End  /  total jobs
Avg Delay        = mean(Delay_Hours) for late jobs only
Breakdown        = by Department, by Shift
```

### 2 — Capacity Utilization
Actual output as a proportion of machine capacity — computed at machine level,
not just in aggregate (the aggregate hides bottlenecks).
```
Capacity_Util % = (Actual_Qty / Machine_Capacity) × 100
```

### 3 — Lead Time
Full elapsed time from scheduled start to actual completion — mean, median,
and P95 to capture tail risk.
```
Lead_Time_hrs = (Actual_End − Scheduled_Start).total_seconds() / 3600
P95           = np.quantile(Lead_Time_hrs, 0.95)
```

### 4 — Resource Utilization
Workers deployed as a share of maximum available — by shift and department.
Flat variance across shifts signals a sequencing problem, not a headcount problem.
```
Resource_Util % = (Workers_Assigned / Max_Workers) × 100
```

### 5 — Efficiency Score Distribution
Segmented into four operational bands and analyzed via KDE to identify whether
underperformance is a talent gap (bimodal) or systems gap (normal distribution).

| Band | Range | Finding |
|------|-------|---------|
| Critical | < 50 | 6.8% of jobs |
| Low | 50 – 65 | 27.0% of jobs |
| Moderate | 65 – 80 | 37.6% of jobs |
| High | > 80 | 28.6% of jobs |

---

## 📈 Visualizations

### Viz 1 — Production Schedule vs Actual
Four-panel overview: monthly scheduled vs actual quantity (grouped bar), job
duration scatter with on-time/late coloring, schedule adherence by department,
and delay distribution histogram.

### Viz 2 — Resource Allocation Heatmap
Two heatmaps side by side: Department × Shift resource utilization, and
Machine × Department capacity utilization. Surfaces load imbalances invisible
in aggregate metrics.

### Viz 3 — Efficiency Segmentation
KDE density curves per segment, donut chart of segment share, box plots by
department, and shift-level efficiency bar chart.

### Viz 4 — Delay Root Cause Analysis
Pareto chart with cumulative %, average delay duration per cause, stacked
100% bar by department, and delay hours vs efficiency score scatter with
OLS regression trend line.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Run with the Kaggle dataset
1. Download `manufacturing_production_data.csv` from
   [Kaggle](https://www.kaggle.com/datasets/ziya07/manufacturing-production-data)
2. In `manufacturing_analysis.py`, replace:
```python
# Line ~75 — swap synthetic data for real CSV
df = generate_dataset(500)
```
with:
```python
df = pd.read_csv("manufacturing_production_data.csv")
```
3. Map your column names in the **Derived columns** section (~line 90) if they
   differ from the schema below.

### Run with synthetic data (no download needed)
```bash
python manufacturing_analysis.py
```
The script will generate a realistic 500-job synthetic dataset matching the
Kaggle schema and produce all outputs immediately.

---

## 🗃️ Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `Job_ID` | string | Unique job identifier |
| `Machine_ID` | string | Machine assigned (M001–M020) |
| `Department` | string | Assembly / Machining / Fabrication / Quality / Packaging |
| `Shift` | string | Morning / Afternoon / Night |
| `Scheduled_Start` | datetime | Planned job start |
| `Scheduled_End` | datetime | Planned job end |
| `Actual_Start` | datetime | Real job start |
| `Actual_End` | datetime | Real job end |
| `Scheduled_Qty` | int | Planned output units |
| `Actual_Qty` | int | Actual output units |
| `Machine_Capacity` | int | Max units machine can produce |
| `Workers_Assigned` | int | Workers deployed on job |
| `Max_Workers` | int | Maximum workers available |
| `Efficiency_Score` | float | 0–100 efficiency rating |
| `Delay_Hours` | float | Hours past scheduled start |
| `Delay_Reason` | string | Root cause category |

---

## 💡 Key Takeaways

```
❌ 82% of jobs miss schedule    →  Planning ≠ Execution
⚠️  79.5% capacity util         →  Aggregate hides machine-level imbalance
📉  P95 lead time is 2× mean    →  Variance, not average, is the customer problem
🔄  68.9% resource util (flat)  →  Job sequencing gap, not a shift problem
📊  Normal efficiency curve     →  Systems constraint, not a talent gap
🎯  2 causes = 80% of delays    →  Pareto confirmed — fix is concentrated
```

---

## 📝 Related Content

- 📊 [Kaggle Dataset](https://www.kaggle.com/datasets/ziya07/manufacturing-production-data)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, transformation, groupby aggregations |
| `numpy` | Numerical ops, percentiles, synthetic data generation |
| `matplotlib` | All chart rendering, custom dark theme, GridSpec layouts |
| `seaborn` | Heatmaps, box plots, statistical overlays |
| `scipy.stats` | KDE estimation, OLS regression on delay-efficiency relationship |

---

## 📄 License

MIT — free to use, adapt, and share with attribution.

---

*Built as a demonstration of production analytics engineering using open manufacturing data.*
*Methodology, insights, and visualizations are reproducible end-to-end from a single script.*
