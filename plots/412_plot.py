import os
import json
from collections import defaultdict
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# === CONFIG ===
DATA_DIR = "diffusion_revision_traces"
PLOT_DIR = "results_figures"
os.makedirs(PLOT_DIR, exist_ok=True)

# === LOAD RESULTS ===
results = []
for fname in os.listdir(DATA_DIR):
    if fname.endswith(".json") and not fname.startswith("generation_"):
        with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
            results.append(json.load(f))

print(f"Loaded {len(results)} problem traces.")

# === DATA STRUCTURES ===
feedback_lengths = []
error_type_attempt_success = defaultdict(lambda: defaultdict(int))
error_type_attempt_total = defaultdict(lambda: defaultdict(int))
solved_after_rev = {1: 0, 2: 0, 3: 0}
total_problems = len(results)
num_revisions = 3

# === PROCESS ===
for r in results:
    # Track when problem was solved
    if r["final_correct"]:
        if r["num_revisions"] == 0:
            solved_after_rev[1] += 1  # solved immediately
        elif r["num_revisions"] == 1:
            solved_after_rev[1] += 0
            solved_after_rev[2] += 1
        elif r["num_revisions"] == 2:
            solved_after_rev[3] += 1
        elif r["num_revisions"] == 3:
            solved_after_rev[3] += 1

    for attempt in r.get("revision_attempts_history", []):
        etype = attempt["error_type"].lower()
        if etype not in ["conceptual", "structural", "computational", "interpretation"]:
            etype = "other"
        stage = attempt["revision_stage"] + 1
        error_type_attempt_total[etype][stage] += 1
        if attempt["addressed_feedback"]:
            error_type_attempt_success[etype][stage] += 1
        feedback_lengths.append(len(attempt["revision_text"].split()))

# === SUCCESS RATE HEATMAP DATA ===
heatmap_data = []
for etype in ["conceptual", "computational", "structural", "interpretation", "other"]:
    row = []
    for stage in [1, 2, 3]:
        total = error_type_attempt_total[etype][stage]
        success = error_type_attempt_success[etype][stage]
        rate = (success / total * 100) if total > 0 else 0
        row.append(rate)
    heatmap_data.append(row)

df_heatmap = pd.DataFrame(
    heatmap_data,
    index=["Conceptual", "Computational", "Structural", "Interpretation", "Other"],
    columns=["Revision 1", "Revision 2", "Revision 3"]
)

# === PRINT SUMMARY ===
print(f"\nAverage feedback length (words): {mean(feedback_lengths)/num_revisions:.2f}")
print("\nSuccess rate heatmap data (%):")
print(df_heatmap)

print("\nProportion of problems solved after each revision attempt:")
for rev, count in solved_after_rev.items():
    print(f"After Rev {rev}: {count} ({count/total_problems*100:.2f}%)")

# === PLOT HEATMAP ===
plt.figure(figsize=(7, 5))
sns.heatmap(df_heatmap, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Success Rate (%)'})
plt.title("Error Type × Revision Attempt Success Rates")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "error_type_revision_success_heatmap.png"), dpi=300)
plt.close()