# === PLOTS (Stacked Bar Chart Version - Final Adjustments) ===
import os
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# --- Configuration ---
PLOT_DIR = "results_figures"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Custom Seaborn Style (based on the sample but without vertical grid) ---
sns.set_theme(style="whitegrid", font_scale=1.0) # Slightly smaller font
# Turn off vertical grid lines after setting the theme
plt.rcParams.update({
    "axes.grid.axis": "y",  # Only horizontal grid lines
})

# --- Figure 1: Accuracy by revision step (k) for each dataset (Per Dataset Subplots) ---
# Hardcoded example numbers (replace with your final results)
accuracy_data = {
    "Big-Math": [49.0, 53.6, 56.0, 56.8],
    "GSM8K":    [87.2, 89.4, 90.4, 90.8],
    "MATH":     [45.2, 48.0, 48.8, 49.2],
}

# --- Prepare data for plotting ---
datasets = list(accuracy_data.keys())
original_labels = ["Acc@0", "Acc@1", "Acc@2", "Acc@3"]

# Create DataFrame for original data
plot_data = {"Dataset": datasets}
for i, step_label in enumerate(original_labels):
    plot_data[step_label] = [accuracy_data[ds][i] for ds in datasets]
df_plot = pd.DataFrame(plot_data)

# --- Calculate incremental gains for stacking ---
# Gains: Acc@1 - Acc@0, Acc@2 - Acc@1, Acc@3 - Acc@2
acc_steps_labels_to_plot = ["Acc@1 Gain", "Acc@2 Gain", "Acc@3 Gain"]
incremental_gains_to_plot = {}
incremental_gains_to_plot[acc_steps_labels_to_plot[0]] = df_plot["Acc@1"].values - df_plot["Acc@0"].values
incremental_gains_to_plot[acc_steps_labels_to_plot[1]] = df_plot["Acc@2"].values - df_plot["Acc@1"].values
incremental_gains_to_plot[acc_steps_labels_to_plot[2]] = df_plot["Acc@3"].values - df_plot["Acc@2"].values

# --- Y-axis limits and base values for each subplot ---
# The y-axis will start slightly below Acc@0 and end slightly above Acc@3
# We define a small margin to add below and above.
# --- Y-axis limits and base values for each subplot ---
# The y-axis will start at Acc@0 and end slightly above Acc@3
y_margin = {
    "Big-Math": 2.0,
    "GSM8K": 1.5,
    "MATH": 2.0,
}
custom_ylims = {}
y_bases = {} # Store Acc@0 for each dataset to use as base
for dataset in datasets:
    acc0 = df_plot[df_plot["Dataset"] == dataset]["Acc@0"].iloc[0]
    acc3 = df_plot[df_plot["Dataset"] == dataset]["Acc@3"].iloc[0]
    margin = y_margin[dataset]
    # 🔧 Changed: Lower limit is now exactly Acc@0
    custom_ylims[dataset] = (acc0, acc3 + margin)
    y_bases[dataset] = acc0

# --- Create the figure with subplots ---
fig, axes = plt.subplots(1, len(datasets), figsize=(9, 4.5), sharey=False)
fig.suptitle("Accuracy Gain Breakdown per Dataset", y=0.92, fontsize=14, weight='bold')

# Use a darker sequential palette for the gain segments
step_colors = sns.color_palette("Blues", 6)[2:5] # Pick 3 shades

bar_width = 0.2
x_index = 0

# --- Plot a stacked bar for each dataset ---
for j, dataset in enumerate(datasets):
    ax = axes[j]
    
    # Get values for this specific dataset
    base_value = y_bases[dataset] # This is Acc@0, used as the visual base
    acc1_val = df_plot.loc[j, "Acc@1"]
    acc2_val = df_plot.loc[j, "Acc@2"]
    acc3_val = df_plot.loc[j, "Acc@3"]
    
    gains = [
        incremental_gains_to_plot["Acc@1 Gain"][j],
        incremental_gains_to_plot["Acc@2 Gain"][j],
        incremental_gains_to_plot["Acc@3 Gain"][j],
    ]
    
    # The bottom of the first gain segment is Acc@0
    bottom_value = base_value

    # Plot the gain segments stacked on top of Acc@0
    bars = []
    for i, (gain_label, gain_value) in enumerate(zip(acc_steps_labels_to_plot, gains)):
        bar = ax.bar(
            x_index,
            gain_value,
            bar_width,
            bottom=bottom_value,
            # Label for legend
            label=gain_label.replace(" Gain", "") if " Gain" in gain_label else gain_label,
            color=step_colors[i],
            edgecolor='white',
            linewidth=0.5
        )
        bars.append(bar)
        bottom_value += gain_value

    # --- Labels and formatting for subplot ---
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"{dataset}", fontsize=12)
    ax.set_xticks([x_index])
    ax.set_xticklabels([""])
    
    # Set custom individual y-axis limits, starting from Acc@0
    ax.set_ylim(custom_ylims[dataset])
    
    # --- Add text labels for actual Acc values ---
    # Acc@1 value just under the first gain segment (inside or just below it)
    y_pos_acc1 = base_value + gains[0] * 0.5 # Middle of Acc@1 gain segment
    ax.text(
        x_index, y_pos_acc1,
        f'{acc1_val:.1f}%', ha='center', va='center', fontsize=9, weight='bold', color='white'
    )
    
    # Acc@2 value just under the second gain segment
    y_pos_acc2 = base_value + gains[0] + gains[1] * 0.5 # Middle of Acc@2 gain segment
    ax.text(
        x_index, y_pos_acc2,
        f'{acc2_val:.1f}%', ha='center', va='center', fontsize=9, weight='bold', color='white'
    )
    
    # Acc@3 value on top of the full bar
    ax.text(
        x_index, acc3_val + 0.3,
        f'{acc3_val:.1f}%', ha='center', va='bottom', fontsize=9, weight='bold', color='black'
    )
    
    # Add a subtle line at the Acc@0 level for reference
    ax.axhline(y=base_value, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    
    ax.margins(x=0.5)

# Add a single legend below the plots
handles, labels = axes[0].get_legend_handles_labels()
# Clean labels: Acc@1, Acc@2, Acc@3 (gains)
clean_labels = [label.replace(" Gain", "") for label in acc_steps_labels_to_plot]
fig.legend(handles, clean_labels, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, frameon=False, fontsize=12)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plot_path = os.path.join(PLOT_DIR, "accuracy_by_revision_per_dataset.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(dedent(f"""
    Per-dataset stacked bar plot saved to: {plot_path}

    Notes:
    - Three subplots, one for each dataset.
    - Bars are thin (width=0.2).
    - Y-axis for each starts just below Acc@0 and ends above Acc@3.
    - Acc@0 is the visual base; no gray segment anymore.
    - Actual Acc@1 and Acc@2 values are labeled inside their gain segments.
    - Final Acc@3 value is labeled on top of the full bar.
    - A dashed line marks the Acc@0 level for reference.
    - A single legend below explains the gain segments (Acc@1, Acc@2, Acc@3).
"""))