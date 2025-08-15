import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
PLOT_DIR = "results_figures"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Custom Seaborn Style (based on the sample but without vertical grid) ---
sns.set_theme(style="whitegrid", font_scale=1.2)  # Increased font scale
# Turn off vertical grid lines after setting the theme
plt.rcParams.update({
    "axes.grid.axis": "y",  # Only horizontal grid lines
})

# Max revision budget values
ks = list(range(0, 11))

# Pooled metrics for SFT + DPO across all datasets (n = 1500)
accuracy = [62.0, 64.6, 65.0, 65.4, 65.5, 65.6, 65.6, 65.6, 65.5, 65.5, 65.4]
degradation = [0.0, 1.8, 2.0, 2.1, 2.1, 2.1, 2.2, 2.2, 2.3, 2.3, 2.4]
termination = [0.0, 84.1, 90.1, 92.4, 92.4, 92.6, 92.6, 92.6, 92.4, 92.3, 92.2]

# Create figure with vertically stacked subplots - decreased width
fig, (ax1_main, ax2_deg) = plt.subplots(2, 1, figsize=(7, 9), dpi=300)
fig.suptitle("Revision Budget Analysis (SFT + DPO)", fontsize=16, weight='bold', y=0.96)

# Use updated colors
accuracy_color = 'green'  # Changed to green
termination_color = 'blue'  # Changed to blue
degradation_color = '#d62728'  # Keep red for degradation

# ========== TOP SUBPLOT: Accuracy and Termination ==========
# Left y-axis for accuracy
acc_line, = ax1_main.plot(
    ks,
    accuracy,
    color=accuracy_color,
    marker="o",
    markersize=7,
    linewidth=3,
    label="Final Accuracy",
    zorder=10
)

# Set zoomed y-limits for accuracy to show variation
ax1_main.set_ylim(61, 67)
ax1_main.set_xlabel("Max Revisions", fontsize=14)
ax1_main.set_ylabel("Final Accuracy (%)", fontsize=14, color=accuracy_color)
ax1_main.tick_params(axis="y", labelcolor=accuracy_color, labelsize=12)
ax1_main.tick_params(axis="x", labelsize=12)
ax1_main.set_xticks(ks)
ax1_main.set_title("Accuracy and Termination Rate", fontsize=14, pad=10)

# Set specific y-ticks for left axis to ensure alignment
ax1_main.set_yticks(np.arange(61, 68, 1))

# Right y-axis for termination
ax1_term = ax1_main.twinx()

# For termination, we'll plot from k=1 onwards since k=0 has 0% termination
# But we'll extend the line to k=0 visually
term_line, = ax1_term.plot(
    ks[1:],
    termination[1:],
    color=termination_color,
    marker="^",
    markersize=7,
    linewidth=3,
    label="Termination Rate",
    zorder=5
)

# Add a dashed line from (0, 0) to (1, termination[1]) to show it comes from 0
ax1_term.plot(
    [0, 1],
    [0, termination[1]],
    color=termination_color,
    linewidth=3,
    linestyle='--',
    alpha=0.7,
    zorder=5
)

# Set zoomed y-limits for termination to show variation
ax1_term.set_ylim(75, 96)
ax1_term.set_ylabel("Termination Rate (%)", fontsize=14, color=termination_color)
ax1_term.tick_params(axis="y", labelcolor=termination_color, labelsize=12)

# Set specific y-ticks for right axis to align with left axis grid
# We want 7 ticks to match the left axis (61-67 = 7 values)
ax1_term.set_yticks(np.linspace(75, 96, 7))

# ========== BOTTOM SUBPLOT: Degradation Only ==========
deg_line_full, = ax2_deg.plot(
    ks,
    degradation,
    color=degradation_color,
    marker="s",
    markersize=7,
    linewidth=3,
    label="Degradation Rate",
)

ax2_deg.set_ylim(0, 5)
ax2_deg.set_xlabel("Max Revisions", fontsize=14)
ax2_deg.set_ylabel("Degradation Rate (%)", fontsize=14, color=degradation_color)
ax2_deg.tick_params(axis="y", labelcolor=degradation_color, labelsize=12)
ax2_deg.tick_params(axis="x", labelsize=12)
ax2_deg.set_xticks(ks)
ax2_deg.set_title("Degradation Rate", fontsize=14, pad=10)

# Set y-ticks for degradation subplot
ax2_deg.set_yticks(np.arange(0, 6, 1))

# Build legends for each subplot
# Top subplot legend
lines1 = [acc_line, term_line]
labels1 = [l.get_label() for l in lines1]
leg1 = ax1_main.legend(
    lines1, 
    labels1, 
    loc="lower right", 
    frameon=False,
    fontsize=12,
    ncol=2,
)

# Bottom subplot legend
leg2 = ax2_deg.legend(
    loc="upper left", 
    frameon=False,
    fontsize=12,
)

# Adjust layout with much more vertical spacing
plt.tight_layout()
plt.subplots_adjust(hspace=0.35)  # Increased from 0.3 to 0.45

# Save with high DPI and tight bounding box
plot_path = os.path.join(PLOT_DIR, "revision_budget_stacked.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()