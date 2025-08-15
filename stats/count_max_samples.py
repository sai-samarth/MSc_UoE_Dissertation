import json
import re
from datasets import load_dataset

# Load dataset
dataset = load_dataset("saital/llama3_8b_self_revision_sft_50K", split="train")

# Parse dialogue if needed
if isinstance(dataset[0]['dialogue'], str):
    dataset = dataset.map(lambda ex: {**ex, "dialogue": json.loads(ex["dialogue"])})

# Filter: final_correct == True, last message assistant, contains [NO_REV]
def is_good_trace_no_boxed(ex):
    if not ex.get("final_correct", False):
        return False
    dlg = ex.get("dialogue", [])
    if not dlg or dlg[-1].get("role") != "assistant":
        return False
    last_msg = dlg[-1].get("content", "")
    if "[NO_REV]" not in last_msg:
        return False
    return True

# dataset = dataset.filter(is_good_trace_no_boxed)

print(f"Total valid traces after filter: {len(dataset)}")

# Count available samples per revision count
rev_counts = {}
for ex in dataset:
    rev = ex["num_revisions"]
    rev_counts[rev] = rev_counts.get(rev, 0) + 1

print("\nAvailable valid samples by revision count:")
for rev in sorted(rev_counts.keys()):
    print(f"  {rev} revisions: {rev_counts[rev]}")

# Target distributions
mixed_dist = {0: 0.30, 1: 0.25, 2: 0.25, 3: 0.20}
curriculum_phases = {
    1: {0: 0.50, 1: 0.50},
    2: {0: 0.20, 1: 0.40, 2: 0.40},
    3: {0: 0.10, 1: 0.20, 2: 0.30, 3: 0.40},
    4: {0: 0.35, 1: 0.25, 2: 0.25, 3: 0.15}
}

# Function: max total for a single distribution
def max_total_for_distribution(dist, available_counts):
    max_totals = []
    for rev, ratio in dist.items():
        if rev not in available_counts or available_counts[rev] == 0:
            return 0
        max_totals.append(available_counts[rev] / ratio)
    return int(min(max_totals))

# Function: sequential allocation for curriculum
def max_total_curriculum_sequential(phases, available_counts):
    counts = available_counts.copy()
    total_used = 0
    for phase, dist in phases.items():
        max_phase_total = int(min(counts[rev] / ratio for rev, ratio in dist.items()))
        for rev, ratio in dist.items():
            use_count = int(max_phase_total * ratio)
            counts[rev] -= use_count
        total_used += max_phase_total
    return total_used

# Compute max totals
max_total_mixed = max_total_for_distribution(mixed_dist, rev_counts)
max_total_curriculum_seq = max_total_curriculum_sequential(curriculum_phases, rev_counts)

print(f"\nMax total samples for Mixed approach: {max_total_mixed}")
print(f"Max total samples for Curriculum approach (sequential actual): {max_total_curriculum_seq}")