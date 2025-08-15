import json
import os
from collections import Counter
from statistics import mean
from datasets import Dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# === CONFIG ===
SFT_PATH = "finetuning_dataset.jsonl"
DPO_PATH = "dpo_dataset.jsonl"
TOKENIZER_NAME = "meta-llama/Meta-Llama-3.1-8B"  # Change if needed
PLOT_DIR = "results_figures"
CACHE_FILE = "dataset_stats_cache.pkl"
os.makedirs(PLOT_DIR, exist_ok=True)

# === LOAD TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Try to load cached stats
if os.path.exists(CACHE_FILE):
    print("Loading cached statistics...")
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    sft_data_len = cache['sft_data_len']
    rev_dist = cache['rev_dist']
    rev_dist_split = cache['rev_dist_split']
    error_type_counts = cache['error_type_counts']
    error_type_counts_clean = cache['error_type_counts_clean']
    final_correct_count = cache['final_correct_count']
    final_conf_count = cache['final_conf_count']
    conv_lengths_turns = cache['conv_lengths_turns']
    conv_lengths_tokens = cache['conv_lengths_tokens']
    success_by_revision = cache['success_by_revision']
    total_by_revision = cache['total_by_revision']
    dpo_data_len = cache['dpo_data_len']
    chosen_rev_dist = cache['chosen_rev_dist']
    chosen_rev_dist_split = cache['chosen_rev_dist_split']
    rejection_reason_counts = cache['rejection_reason_counts']
    prompt_token_lengths = cache['prompt_token_lengths']
    chosen_token_lengths = cache['chosen_token_lengths']
    rejected_token_lengths = cache['rejected_token_lengths']
else:
    print("Computing statistics from scratch...")
    
    # === LOAD SFT DATASET ===
    print("Loading SFT dataset...")
    sft_data = [json.loads(line) for line in open(SFT_PATH, "r", encoding="utf-8")]
    sft_data_len = len(sft_data)
    print(f"Total SFT examples: {sft_data_len}")

    # === SFT STATS ===
    rev_dist = Counter()
    rev_dist_split = Counter()
    error_type_counts = Counter()
    error_type_counts_clean = Counter()
    final_correct_count = 0
    final_conf_count = 0
    conv_lengths_turns = []
    conv_lengths_tokens = []

    # Error type mapping for cleaning
    error_type_map = {
        "interpretation": "Interpretation",
        "conceptual": "Conceptual",
        "structural": "Structural",
        "computational": "Computational"
    }

    # Track success by revision number for accuracy calculation
    success_by_revision = {0: 0, 1: 0, 2: 0, 3: 0}
    total_by_revision = {0: 0, 1: 0, 2: 0, 3: 0}

    for ex in sft_data:
        num_rev = ex["num_revisions"]
        rev_dist[num_rev] += 1
        total_by_revision[num_rev] += 1
        
        if ex["final_correct"]:
            success_by_revision[num_rev] += 1
        
        # Split max revisions into solved vs failed
        if ex["num_revisions"] == 3:
            if ex["final_correct"]:
                rev_dist_split["3_solved"] += 1
            else:
                rev_dist_split["3_failed"] += 1
        else:
            rev_dist_split[str(ex["num_revisions"])] += 1
        
        if ex["final_correct"]:
            final_correct_count += 1
        if ex.get("has_final_confirmation", False):
            final_conf_count += 1
        
        # Original error types
        for et in ex.get("error_types_addressed", []):
            error_type_counts[et] += 1
        
        # Cleaned error types
        for et in ex.get("error_types_addressed", []):
            et_lower = et.lower()
            if et_lower in error_type_map:
                error_type_counts_clean[error_type_map[et_lower]] += 1
            else:
                error_type_counts_clean["Other"] += 1
        
        conv_lengths_turns.append(len(ex["dialogue"]))
        text = " ".join([m["content"] for m in ex["dialogue"]])
        conv_lengths_tokens.append(len(tokenizer.encode(text)))

    # === LOAD DPO DATASET ===
    print("Loading DPO dataset...")
    dpo_data = [json.loads(line) for line in open(DPO_PATH, "r", encoding="utf-8")]
    dpo_data_len = len(dpo_data)
    print(f"Total DPO pairs: {dpo_data_len}")

    # === DPO STATS ===
    chosen_rev_dist = Counter()
    chosen_rev_dist_split = Counter()
    rejection_reason_counts = Counter()
    prompt_token_lengths = []
    chosen_token_lengths = []
    rejected_token_lengths = []

    for ex in dpo_data:
        meta = ex.get("metadata", {})
        revs = meta.get("chosen_revisions", -1)
        chosen_rev_dist[revs] += 1
        
        # Split for DPO (no final_correct info, so just group by 3)
        if revs == 3:
            chosen_rev_dist_split["3_revs"] += 1
        else:
            chosen_rev_dist_split[str(revs)] += 1
        
        rejection_reason_counts[meta.get("rejection_reason", "unknown")] += 1
        
        prompt_token_lengths.append(len(tokenizer.encode(" ".join([m["content"] for m in ex["prompt"]]))))
        chosen_token_lengths.append(len(tokenizer.encode(" ".join([m["content"] for m in ex["chosen"]]))))
        rejected_token_lengths.append(len(tokenizer.encode(" ".join([m["content"] for m in ex["rejected"]]))))

    # Save cache
    cache = {
        'sft_data_len': sft_data_len,
        'rev_dist': rev_dist,
        'rev_dist_split': rev_dist_split,
        'error_type_counts': error_type_counts,
        'error_type_counts_clean': error_type_counts_clean,
        'final_correct_count': final_correct_count,
        'final_conf_count': final_conf_count,
        'conv_lengths_turns': conv_lengths_turns,
        'conv_lengths_tokens': conv_lengths_tokens,
        'success_by_revision': success_by_revision,
        'total_by_revision': total_by_revision,
        'dpo_data_len': dpo_data_len,
        'chosen_rev_dist': chosen_rev_dist,
        'chosen_rev_dist_split': chosen_rev_dist_split,
        'rejection_reason_counts': rejection_reason_counts,
        'prompt_token_lengths': prompt_token_lengths,
        'chosen_token_lengths': chosen_token_lengths,
        'rejected_token_lengths': rejected_token_lengths
    }
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved to {CACHE_FILE}")

# === PRINT SUMMARY ===
print("\n=== SFT DATASET STATS ===")
print(f"Revision distribution: {dict(rev_dist)}")
print(f"Solved/Fail split (revisions): {dict(rev_dist_split)}")
print(f"Final correct %: {final_correct_count / sft_data_len * 100:.2f}%")
print(f"Error type counts: {dict(error_type_counts)}")
print(f"Cleaned error type counts: {dict(error_type_counts_clean)}")
print(f"Has final confirmation %: {final_conf_count / sft_data_len * 100:.2f}%")
print(f"Avg conversation length (turns): {mean(conv_lengths_turns):.2f}")
print(f"Avg conversation length (tokens): {mean(conv_lengths_tokens):.2f}")

print("\n=== DPO DATASET STATS ===")
print(f"Chosen revisions distribution: {dict(chosen_rev_dist)}")
print(f"Chosen revisions split: {dict(chosen_rev_dist_split)}")
print(f"Rejection reason counts: {dict(rejection_reason_counts)}")
print(f"Avg prompt length (tokens): {mean(prompt_token_lengths):.2f}")
print(f"Avg chosen length (tokens): {mean(chosen_token_lengths):.2f}")
print(f"Avg rejected length (tokens): {mean(rejected_token_lengths):.2f}")

# === PLOTS ===
sns.set_theme(style="whitegrid", font_scale=1.2)

# Figure 1: Revision Distribution with Split
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})

# Create secondary y-axis for percentage
ax2 = ax1.twinx()

# Turn off all grids first
ax1.grid(False)
ax2.grid(False)

# Define green + red color scheme
green_shades = ['#90EE90', '#32CD32', '#228B22', '#006400']  # Light to dark green
red_color = '#8B0000'  # Dark red for unsolved

# --- SFT split with accuracy stacked bar ---
# First, plot the regular bars with green shades + red
sft_order = ["0", "1", "2", "3_solved", "3_failed"]
sft_labels = ["0", "1", "2", "3\n(Solved)", "3\n(Failed)"]
sft_vals = [rev_dist_split.get(k, 0) for k in sft_order]
sft_colors = green_shades + [red_color]  # 4 greens + 1 red

# Calculate bar positions
x_positions = np.arange(len(sft_labels))
bar_width = 0.8

# Draw grid manually before plotting bars
ax1.yaxis.grid(True, alpha=0.3, zorder=1)
ax1.xaxis.grid(False)
ax1.set_axisbelow(True)

# Plot regular bars on primary y-axis
bars = ax1.bar(x_positions, sft_vals, bar_width, color=sft_colors, zorder=3, edgecolor='black', linewidth=0.5)

# Add vertical dotted line before the stacked accuracy bar
ax1.axvline(x=len(sft_labels) - 0.4, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

# Calculate accuracies based on your provided numbers
total_examples = sft_data_len  # 49212
acc_0 = success_by_revision[0] / sft_data_len * 100  # 34.72%
acc_1 = success_by_revision[1] / sft_data_len * 100  # 19.36%
acc_2 = success_by_revision[2] / sft_data_len * 100  # 11.34%
acc_3 = success_by_revision[3] / sft_data_len * 100  # 5.01%
unsolved = (sft_data_len - final_correct_count) / sft_data_len * 100  # 29.55%

# Add stacked accuracy bar on secondary y-axis
stack_x = len(sft_labels) + 0.15
stack_width = bar_width

# Create stacked bars on secondary y-axis with matching green + red colors
p1 = ax2.bar(stack_x, acc_0, stack_width, label='Acc@0', color=green_shades[0], zorder=3, edgecolor='black', linewidth=0.5)
p2 = ax2.bar(stack_x, acc_1, stack_width, bottom=acc_0, label='Acc@1', color=green_shades[1], zorder=3, edgecolor='black', linewidth=0.5)
p3 = ax2.bar(stack_x, acc_2, stack_width, bottom=acc_0+acc_1, label='Acc@2', color=green_shades[2], zorder=3, edgecolor='black', linewidth=0.5)
p4 = ax2.bar(stack_x, acc_3, stack_width, bottom=acc_0+acc_1+acc_2, label='Acc@3', color=green_shades[3], zorder=3, edgecolor='black', linewidth=0.5)
p5 = ax2.bar(stack_x, unsolved, stack_width, bottom=acc_0+acc_1+acc_2+acc_3, label='Unsolved', color=red_color, zorder=3, edgecolor='black', linewidth=0.5)

# Add marginal accuracy labels inside bars (all white)
def add_marginal_label(ax, x, y_center, value):
    if value > 2:  # Only show if segment is large enough
        ax.text(x, y_center, f'{value:.1f}%', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=12, zorder=4)

# Add cumulative accuracy labels on the right with more space
def add_cumulative_label(ax, x, y, value):
    ax.text(x + stack_width/2 + 0.15, y, f'{value:.1f}%', ha='left', va='center', 
            fontsize=12, fontweight='bold', zorder=4)

# Marginal labels (inside bars - all white)
add_marginal_label(ax2, stack_x, acc_0/2, acc_0)
add_marginal_label(ax2, stack_x, acc_0 + acc_1/2, acc_1)
add_marginal_label(ax2, stack_x, acc_0 + acc_1 + acc_2/2, acc_2)
add_marginal_label(ax2, stack_x, acc_0 + acc_1 + acc_2 + acc_3/2, acc_3)
add_marginal_label(ax2, stack_x, acc_0 + acc_1 + acc_2 + acc_3 + unsolved/2, unsolved)

# Cumulative labels (outside bars)
add_cumulative_label(ax2, stack_x, acc_0, acc_0)
add_cumulative_label(ax2, stack_x, acc_0 + acc_1, acc_0 + acc_1)
add_cumulative_label(ax2, stack_x, acc_0 + acc_1 + acc_2, acc_0 + acc_1 + acc_2)
add_cumulative_label(ax2, stack_x, acc_0 + acc_1 + acc_2 + acc_3, acc_0 + acc_1 + acc_2 + acc_3)
add_cumulative_label(ax2, stack_x, 100, 100.0)

# Update x-axis
new_labels = sft_labels + ['Accuracy\nBreakdown']
ax1.set_xticks(list(x_positions) + [stack_x])
ax1.set_xticklabels(new_labels)

# Formatting
ax1.set_title("SFT Revision Distribution and Accuracy Analysis")
ax1.set_xlabel("Revision Outcome")
ax1.set_ylabel("Count", fontsize=14)
ax2.set_ylabel("Percentage (%)", fontsize=14)

# Align y-axes
max_count = max(sft_vals) * 1.1
ax1.set_ylim(0, max_count)
# Scale percentage axis proportionally
ax2.set_ylim(0, 110)

# Add extra space on the right for cumulative labels
ax1.set_xlim(-0.5, stack_x + 1.4)

# Create custom legend elements for percentage labels
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Custom legend elements
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Marginal %', 
           markerfacecolor='white', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Cumulative %', 
           markerfacecolor='black', markeredgecolor='black', markersize=8)
]

# Add first legend for stacked bars (below plot)
first_legend = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), 
                         ncol=6, fontsize=14, frameon=False)

# Add second legend for percentage labels (right side)
# second_legend = ax2.legend(handles=legend_elements, loc='center left', 
#                           bbox_to_anchor=(1.1, 0.5), fontsize=11, frameon=True,
#                           title='Percentage Labels', title_fontsize=12)

# Add the first legend back
ax2.add_artist(first_legend)

# --- DPO split ordering and labels ---
dpo_order_keys = ["0", "1", "2", "3"]
def map_key(k):
    return "3" if k == "3_revs" else str(k)

dpo_agg = {k: 0 for k in dpo_order_keys}
for k, v in chosen_rev_dist_split.items():
    mk = map_key(k)
    if mk in dpo_agg:
        dpo_agg[mk] += v

dpo_labels = dpo_order_keys
dpo_vals = [dpo_agg[k] for k in dpo_order_keys]

sns.barplot(x=dpo_labels, y=dpo_vals, ax=ax3, palette="Greens_d", order=dpo_labels)
ax3.set_title("DPO Chosen Revision Distribution")
ax3.set_xlabel("Number of Revisions")
ax3.set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "revision_distribution_split.png"), dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Error Type Frequency (Cleaned)
plt.figure(figsize=(6, 4))
err_order = ["Interpretation", "Conceptual", "Structural", "Computational", "Other"]
err_vals = [error_type_counts_clean.get(k, 0) for k in err_order]
ax_err = sns.barplot(x=err_order, y=err_vals, palette="Oranges_d", order=err_order)
plt.title("Error Type Frequency in SFT Dataset")
plt.xlabel("Error Type")
plt.ylabel("Count")
plt.xticks(rotation=15, ha="center")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "error_type_frequency_clean.png"), dpi=300)
plt.close()

# Table 1: Summary Stats
summary_table = {
    "SFT Total Examples": sft_data_len,
    "SFT Final Correct %": round(final_correct_count / sft_data_len * 100, 2),
    "SFT Final Confirmation %": round(final_conf_count / sft_data_len * 100, 2),
    "SFT Avg Turns": round(mean(conv_lengths_turns), 2),
    "SFT Avg Tokens": round(mean(conv_lengths_tokens), 2),
    "DPO Total Pairs": dpo_data_len,
    "DPO Avg Prompt Tokens": round(mean(prompt_token_lengths), 2),
    "DPO Avg Chosen Tokens": round(mean(chosen_token_lengths), 2),
    "DPO Avg Rejected Tokens": round(mean(rejected_token_lengths), 2)
}

print("\n=== SUMMARY TABLE ===")
for k, v in summary_table.items():
    print(f"{k}: {v}")

# Print accuracy breakdown for verification
print("\n=== ACCURACY BREAKDOWN ===")
print(f"Acc@0 (marginal): {acc_0:.2f}%")
print(f"Acc@1 (marginal): {acc_1:.2f}%")
print(f"Acc@2 (marginal): {acc_2:.2f}%")
print(f"Acc@3 (marginal): {acc_3:.2f}%")
print(f"Unsolved: {unsolved:.2f}%")
print(f"Total: {acc_0 + acc_1 + acc_2 + acc_3 + unsolved:.2f}%")