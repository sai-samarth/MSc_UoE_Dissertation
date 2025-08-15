import os
import re
import json
import math
import random
import logging
from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "diffusion_revision_traces"
DATA_PATH = "data"
TRAIN_JSON = os.path.join(DATA_PATH, "train_problems.json")
OLD_TEST_JSON = os.path.join(DATA_PATH, "test_problems.json")
FRESH_TEST_JSON = os.path.join(DATA_PATH, "fresh_test_500.json")
SFT_PATH = "finetuning_dataset.jsonl"
DPO_PATH = "dpo_dataset.jsonl"
PLOT_DIR = "results_figures"
SUMMARY_JSON = os.path.join(PLOT_DIR, "data_generation_summary.json")
TRACE_QUALITY_CSV = os.path.join(PLOT_DIR, "trace_quality_summary.csv")

# HF dataset and columns
HF_DATASET = "SynthLabsAI/Big-Math-RL-Verified"
SOLVE_RATE_COL = "llama8b_solve_rate"  # expected in the dataset
ANSWER_COL = "answer"  # raw dataset column name
PROBLEM_COL = "problem"

# Difficulty filter (match your generation defaults)
MIN_SOLVE_RATE = float(os.getenv("MIN_SOLVE_RATE", "0.2"))
MAX_SOLVE_RATE = float(os.getenv("MAX_SOLVE_RATE", "0.7"))

# Tokenizer (for token counts)
TOKENIZER_NAME = "meta-llama/Meta-Llama-3.1-8B"

# Fresh test size
FRESH_TEST_SIZE = 500
SEED = 42

# Plot style
sns.set_theme(style="whitegrid", font_scale=1.2)

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("analysis_4_1")


# -----------------------
# HELPERS
# -----------------------
def ensure_dirs() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # Normalize whitespace, line breaks, some LaTeX spacing variants
    s_norm = s.replace("\r\n", "\n").replace("\r", "\n")
    s_norm = re.sub(r"\s+", " ", s_norm).strip()
    return s_norm


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def has_boxed_answer(text: str) -> bool:
    if not text:
        return False
    return bool(BOXED_RE.search(text))


def extract_boxed_answers(text: str) -> List[str]:
    if not text:
        return []
    return BOXED_RE.findall(text)


def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def problem_key_map(df: pd.DataFrame) -> Dict[str, Dict]:
    """Build a lookup by normalized problem text -> row dict."""
    mapping = {}
    for _, row in df.iterrows():
        key = normalize_text(str(row.get(PROBLEM_COL, "")))
        if key and key not in mapping:
            mapping[key] = {
                "solve_rate": row.get(SOLVE_RATE_COL, None),
                "problem": row.get(PROBLEM_COL, ""),
                "answer": str(row.get(ANSWER_COL, "")),
            }
    return mapping


def tokenize_len(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


def iter_assistant_msgs(dialogue: List[Dict]) -> List[str]:
    return [m["content"] for m in dialogue if m.get("role") == "assistant"]


def iter_user_msgs(dialogue: List[Dict]) -> List[str]:
    return [m["content"] for m in dialogue if m.get("role") == "user"]


def extract_teacher_feedback_texts(dialogue: List[Dict]) -> List[str]:
    """Extract teacher feedback from 'user' messages that start with 'Teacher feedback:'."""
    texts = []
    for m in dialogue:
        if m.get("role") == "user":
            content = m.get("content", "")
            if content.startswith("Teacher feedback:"):
                # get text after the colon
                fb = content[len("Teacher feedback:") :].strip()
                if fb:
                    texts.append(fb)
    return texts


# -----------------------
# 1) LOAD BIG-MATH DATASET, DEDUP, FILTER
# -----------------------
def load_big_math() -> Tuple[pd.DataFrame, Dict]:
    """
    Try to load SynthLabsAI/Big-Math-RL-Verified using datasets.load_dataset.
    If that fails (e.g., due to an incompatible local 'datasets' version), attempt
    a force_redownload. If still failing, fall back to huggingface_hub and try to
    download a sensible raw file (.jsonl/.json/.parquet/.csv) and load with pandas.
    Returns (df_filtered, meta).
    """
    logger.info("Loading Big‑Math dataset via datasets.load_dataset...")
    meta = {}
    try:
        ds = load_dataset(HF_DATASET, split="train")
        df = pd.DataFrame(ds)
    except Exception as e:
        logger.warning(f"load_dataset() failed: {e}")
        logger.info("Retrying with download_mode='force_redownload' ...")
        try:
            ds = load_dataset(HF_DATASET, split="train", download_mode="force_redownload")
            df = pd.DataFrame(ds)
        except Exception as e2:
            logger.warning(f"force_redownload also failed: {e2}")
            # Fallback via huggingface_hub
            logger.info("Falling back to huggingface_hub to find raw files in the repo...")
            try:
                api = HfApi()
                repo_files = api.list_repo_files(HF_DATASET)
            except Exception as e3:
                logger.error(f"Could not list repo files via huggingface_hub: {e3}")
                raise RuntimeError(
                    "Unable to load dataset via `datasets` and huggingface_hub. "
                    "Please upgrade the `datasets` package (pip install -U datasets) "
                    "or provide the Big‑Math dataset locally."
                ) from e3

            # Prefer train-containing files and prefer jsonl/json/parquet/csv in that order
            preferred_exts = [".jsonl", ".json", ".parquet", ".csv"]
            candidate = None
            for ext in preferred_exts:
                # first try files that mention 'train' and have the extension
                cand = [f for f in repo_files if f.lower().endswith(ext) and "train" in f.lower()]
                if cand:
                    candidate = cand[0]
                    break
            if candidate is None:
                # fallback to any file with preferred ext
                for ext in preferred_exts:
                    cand = [f for f in repo_files if f.lower().endswith(ext)]
                    if cand:
                        candidate = cand[0]
                        break

            if candidate is None:
                # nothing suitable
                logger.error("No suitable raw data file (.jsonl/.json/.parquet/.csv) found in HF repo.")
                raise RuntimeError(
                    "Could not find a raw dataset file in the HF repo. "
                    "Please upgrade the `datasets` package or supply the dataset file locally."
                )

            logger.info(f"Downloading candidate file from repo: {candidate}")
            try:
                local_path = hf_hub_download(repo_id=HF_DATASET, filename=candidate)
            except Exception as e4:
                logger.error(f"hf_hub_download failed: {e4}")
                raise

            # Load the downloaded file with pandas based on extension
            if candidate.endswith(".jsonl"):
                df = pd.read_json(local_path, lines=True)
            elif candidate.endswith(".json"):
                df = pd.read_json(local_path)
            elif candidate.endswith(".parquet"):
                df = pd.read_parquet(local_path)
            elif candidate.endswith(".csv"):
                df = pd.read_csv(local_path)
            else:
                raise RuntimeError("Downloaded file has unexpected extension; cannot load.")

    # Post-processing to compute dedup stats and filtering
    n_raw = len(df)
    df = df.rename(columns={ANSWER_COL: "answer"} ) if ANSWER_COL in df.columns else df
    df_dedup = df.drop_duplicates(subset=[PROBLEM_COL], keep="first")
    n_dedup = len(df_dedup)
    dup_removed = n_raw - n_dedup
    dup_pct = (dup_removed / n_raw * 100.0) if n_raw > 0 else 0.0

    df_dedup["problem_norm"] = df_dedup[PROBLEM_COL].map(normalize_text)
    df_norm_dedup = df_dedup.drop_duplicates(subset=["problem_norm"], keep="first")
    n_norm_dedup = len(df_norm_dedup)
    dup_removed_norm = n_raw - n_norm_dedup
    dup_norm_pct = (dup_removed_norm / n_raw * 100.0) if n_raw > 0 else 0.0

    meta.update({
        "n_raw": n_raw,
        "n_dedup_problem": n_dedup,
        "duplicates_removed_problem": dup_removed,
        "duplicates_removed_problem_pct": round(dup_pct, 2),
        "n_dedup_problem_norm": n_norm_dedup,
        "duplicates_removed_problem_norm": dup_removed_norm,
        "duplicates_removed_problem_norm_pct": round(dup_norm_pct, 2),
    })

    # Difficulty filter (if solve rate exists)
    if SOLVE_RATE_COL in df_dedup.columns:
        df_filtered = df_dedup[
            (df_dedup[SOLVE_RATE_COL] >= MIN_SOLVE_RATE) &
            (df_dedup[SOLVE_RATE_COL] <= MAX_SOLVE_RATE)
        ].copy()
        meta.update({
            "n_filtered_by_solve_rate": len(df_filtered),
            "filter_range": [MIN_SOLVE_RATE, MAX_SOLVE_RATE],
        })
    else:
        logger.warning(f"No '{SOLVE_RATE_COL}' in dataset. Skipping difficulty filter.")
        df_filtered = df_dedup.copy()

    return df_filtered, meta
# -----------------------
# 2) RECONSTRUCT DIFFICULTY FOR TRAIN/OLD TEST, BUILD FRESH TEST 500
# -----------------------
def recover_solve_rates_for_json(
    big_math_map: Dict[str, Dict], examples: List[Dict]
) -> List[float]:
    rates = []
    missing = 0
    for ex in examples:
        key = normalize_text(ex.get("problem", ""))
        info = big_math_map.get(key)
        if info and info.get("solve_rate") is not None:
            rates.append(float(info["solve_rate"]))
        else:
            missing += 1
    if missing > 0:
        logger.warning(
            f"Solve rate recovery: {missing}/{len(examples)} not matched in Big‑Math."
        )
    return rates


def write_fresh_test_500(
    df_big_math: pd.DataFrame,
    train_examples: List[Dict],
    out_path: str,
    seed: int = SEED,
    size: int = FRESH_TEST_SIZE,
) -> List[Dict]:
    train_keys = {normalize_text(ex.get("problem", "")) for ex in train_examples}
    df_cand = df_big_math.copy()
    df_cand["key"] = df_cand[PROBLEM_COL].map(normalize_text)
    df_cand = df_cand[~df_cand["key"].isin(train_keys)].copy()

    if len(df_cand) < size:
        logger.warning(
            f"Not enough candidates ({len(df_cand)}) to sample {size}. "
            "Will sample as many as available."
        )
        size = len(df_cand)

    rng = random.Random(seed)
    indices = list(df_cand.index)
    rng.shuffle(indices)
    pick = indices[:size]
    sub = df_cand.loc[pick]

    fresh = []
    for i, (_, row) in enumerate(sub.iterrows(), start=1):
        fresh.append(
            {
                "problem_id": f"fresh_{i}",
                "problem": row.get(PROBLEM_COL, ""),
                "expected_answer": str(row.get(ANSWER_COL, "")),
                "solve_rate": float(row.get(SOLVE_RATE_COL, float("nan"))),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fresh, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Fresh test set written to {out_path} (n={len(fresh)}), "
        "isolated from train by problem text."
    )
    return fresh


# -----------------------
# 3) LOAD RESULTS/SFT/DPO AND COMPUTE STATS
# -----------------------
def load_generation_results() -> List[Dict]:
    if not os.path.exists(DATA_DIR):
        logger.warning(
            f"Results dir '{DATA_DIR}' not found. Trace-quality stats may be limited."
        )
        return []
    results = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".json") and not fname.startswith("generation_"):
            with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
                results.append(json.load(f))
    logger.info(f"Loaded {len(results)} problem traces.")
    return results


def compute_trace_quality(
    results: List[Dict], sft_examples: List[Dict]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build a trace-quality breakdown using available signals.
    Categories:
      - accepted_for_sft: included in finetuning_dataset.jsonl
      - failed_poor_revisions: problem flagged failed_due_to_poor_revisions and no accepted revision
      - missing_boxed_answer: no assistant message contains \\boxed{...}
      - other_unsolved: remaining unsolved not in accepted_for_sft
    """
    sft_ids = set()
    # Create a signature from dialogues to connect? Safer to use problem_id if present.
    for ex in sft_examples:
        pid = ex.get("problem_id")
        if pid:
            sft_ids.add(pid)

    n_total = len(results)
    cats = Counter()

    # Helper: detect any assistant boxed answers
    def has_any_assistant_boxed(dialogue: List[Dict]) -> bool:
        for msg in dialogue:
            if msg.get("role") == "assistant" and has_boxed_answer(msg.get("content", "")):
                return True
        return False

    for r in results:
        pid = r.get("problem_id")
        in_sft = pid in sft_ids
        final_correct = bool(r.get("final_correct", False))
        num_revs = int(r.get("num_revisions", 0))
        failed_poor = bool(r.get("failed_due_to_poor_revisions", False))
        dialogue = r.get("dialogue", [])

        any_boxed = has_any_assistant_boxed(dialogue)

        if in_sft:
            cats["accepted_for_sft"] += 1
        else:
            if failed_poor and num_revs == 0:
                cats["failed_poor_revisions"] += 1
            elif not any_boxed:
                cats["missing_boxed_answer"] += 1
            elif not final_correct:
                cats["other_unsolved"] += 1
            else:
                cats["other"] += 1

    # Build DataFrame
    rows = []
    for k in ["accepted_for_sft", "failed_poor_revisions", "missing_boxed_answer", "other_unsolved", "other"]:
        cnt = cats.get(k, 0)
        pct = (cnt / n_total * 100.0) if n_total > 0 else 0.0
        rows.append({"category": k, "count": cnt, "percent": round(pct, 2)})

    df_quality = pd.DataFrame(rows)
    totals = {
        "total_traces": n_total,
        "accepted_for_sft": cats.get("accepted_for_sft", 0),
        "rejected_total": n_total - cats.get("accepted_for_sft", 0),
    }
    return df_quality, totals


def compute_teacher_feedback_stats(results: List[Dict]) -> Dict:
    lengths_words = []
    for r in results:
        fb_texts = extract_teacher_feedback_texts(r.get("dialogue", []))
        for fb in fb_texts:
            n_words = len(fb.split())
            lengths_words.append(n_words)
    if not lengths_words:
        return {
            "avg_words": 0.0,
            "median_words": 0.0,
            "p95_words": 0.0,
            "count": 0,
        }
    arr = sorted(lengths_words)
    p95_idx = min(len(arr) - 1, int(0.95 * len(arr)))
    return {
        "avg_words": round(mean(arr), 2),
        "median_words": float(arr[len(arr) // 2]),
        "p95_words": float(arr[p95_idx]),
        "count": len(arr),
    }


# -----------------------
# 4) PLOTTING
# -----------------------
def plot_revision_distribution_split(
    sft_examples: List[Dict], out_path: str
) -> None:
    rev_dist_split = Counter()
    for ex in sft_examples:
        n_rev = ex.get("num_revisions", 0)
        if n_rev == 3:
            if ex.get("final_correct", False):
                rev_dist_split["3_solved"] += 1
            else:
                rev_dist_split["3_failed"] += 1
        else:
            rev_dist_split[str(n_rev)] += 1

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    order = ["0", "1", "2", "3_solved", "3_failed"]
    labels = ["0", "1", "2", "3", "Unsolved"]
    vals = [rev_dist_split.get(k, 0) for k in order]
    sns.barplot(x=labels, y=vals, ax=ax, palette="Blues_d", order=labels)
    ax.set_title("SFT Revision Distribution (Solved vs Failed)")
    ax.set_xlabel("Revision Outcome")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_error_type_frequency(
    sft_examples: List[Dict], out_path: str
) -> None:
    # Cleaned error type counts
    map_clean = {
        "interpretation": "Interpretation",
        "conceptual": "Conceptual",
        "structural": "Structural",
        "computational": "Computational",
    }
    counts = Counter()
    for ex in sft_examples:
        for et in ex.get("error_types_addressed", []):
            lab = map_clean.get(str(et).lower(), "Other")
            counts[lab] += 1

    order = ["Interpretation", "Conceptual", "Structural", "Computational", "Other"]
    vals = [counts.get(k, 0) for k in order]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=order, y=vals, palette="Oranges_d", order=order)
    plt.title("Error Type Frequency in SFT Dataset")
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.xticks(rotation=15, ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_error_type_revision_success_heatmap(
    results: List[Dict], out_path: str
) -> None:
    error_type_attempt_success = defaultdict(lambda: defaultdict(int))
    error_type_attempt_total = defaultdict(lambda: defaultdict(int))
    for r in results:
        for attempt in r.get("revision_attempts_history", []):
            et = str(attempt.get("error_type", "other")).lower()
            if et not in [
                "conceptual",
                "computational",
                "structural",
                "interpretation",
            ]:
                et = "other"
            stage = int(attempt.get("revision_stage", 0)) + 1
            error_type_attempt_total[et][stage] += 1
            if attempt.get("addressed_feedback", False):
                error_type_attempt_success[et][stage] += 1

    heatmap_data = []
    idx_labels = [
        "Conceptual",
        "Computational",
        "Structural",
        "Interpretation",
        "Other",
    ]
    col_labels = ["Revision 1", "Revision 2", "Revision 3"]
    for et in ["conceptual", "computational", "structural", "interpretation", "other"]:
        row = []
        for stage in [1, 2, 3]:
            total = error_type_attempt_total[et][stage]
            success = error_type_attempt_success[et][stage]
            rate = (success / total * 100.0) if total > 0 else 0.0
            row.append(rate)
        heatmap_data.append(row)

    df_heat = pd.DataFrame(heatmap_data, index=idx_labels, columns=col_labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        df_heat,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Success Rate (%)"},
    )
    plt.title("Error Type × Revision Attempt Success Rates")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_difficulty_overlay(
    train_rates: List[float],
    old_test_rates: Optional[List[float]],
    fresh_rates: List[float],
    out_path: str,
) -> None:
    data = []
    for r in train_rates:
        data.append({"split": "Train", "solve_rate": r})
    if old_test_rates is not None:
        for r in old_test_rates:
            data.append({"split": "Old Test", "solve_rate": r})
    for r in fresh_rates:
        data.append({"split": "Fresh Test 500", "solve_rate": r})

    if not data:
        logger.warning("No difficulty data available for overlay plot.")
        return

    df = pd.DataFrame(data)
    plt.figure(figsize=(7, 4))
    # Layered histograms
    sns.histplot(
        data=df,
        x="solve_rate",
        hue="split",
        bins=20,
        multiple="layer",
        stat="count",
        palette="Set2",
        edgecolor="white",
    )
    plt.title("Problem Difficulty (Solve‑Rate) Distribution")
    plt.xlabel("Solve Rate")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_trace_quality_breakdown(df_quality: pd.DataFrame, out_path: str) -> None:
    dfq = df_quality.copy()
    # Friendly labels
    label_map = {
        "accepted_for_sft": "Accepted for SFT",
        "failed_poor_revisions": "Failed: Poor Revisions",
        "missing_boxed_answer": "Failed: Missing Boxed Answer",
        "other_unsolved": "Failed: Other Unsolved",
        "other": "Other",
    }
    dfq["label"] = dfq["category"].map(label_map).fillna(dfq["category"])
    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=dfq, x="label", y="count", palette="Purples_d", order=dfq["label"]
    )
    plt.title("Trace Quality Control Breakdown")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=15, ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_teacher_feedback_length_hist(
    fb_stats: Dict, fb_lengths: List[int], out_path: str
) -> None:
    if not fb_lengths:
        logger.info("No teacher feedback lengths to plot.")
        return
    plt.figure(figsize=(7, 4))
    sns.histplot(fb_lengths, bins=30, color="#1f77b4", edgecolor="white")
    plt.title("Teacher Feedback Length (words)")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------
# MAIN
# -----------------------
def main() -> None:
    ensure_dirs()

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as e:
        logger.warning(
            f"Failed to load tokenizer '{TOKENIZER_NAME}'. Token counts may be zero. "
            f"Error: {e}"
        )
        tokenizer = None

    # 1) Big‑Math, dedup, filter
    df_big_math, big_math_meta = load_big_math()
    big_math_map = problem_key_map(df_big_math)  # normalized lookup

    # 2) Load train/old test JSONs (if present), recover difficulty
    train_examples = load_json(TRAIN_JSON)
    old_test_examples = load_json(OLD_TEST_JSON)

    train_rates = (
        recover_solve_rates_for_json(big_math_map, train_examples)
        if train_examples
        else []
    )
    old_test_rates = (
        recover_solve_rates_for_json(big_math_map, old_test_examples)
        if old_test_examples
        else None
    )

    # 3) Build fresh test set of 500 (isolated from train)
    fresh_examples = write_fresh_test_500(
        df_big_math, train_examples, FRESH_TEST_JSON, seed=SEED, size=FRESH_TEST_SIZE
    )
    fresh_rates = [float(ex.get("solve_rate", float("nan"))) for ex in fresh_examples]
    fresh_rates = [r for r in fresh_rates if not math.isnan(r)]

    # 4) Load generation results and SFT/DPO datasets
    results = load_generation_results()
    sft_examples = load_jsonl(SFT_PATH)
    dpo_pairs = load_jsonl(DPO_PATH)

    # 5) Core stats for SFT dataset
    sft_rev_dist = Counter()
    sft_rev_split = Counter()
    sft_error_type_counts = Counter()
    sft_error_type_clean = Counter()
    sft_final_correct = 0
    sft_final_conf = 0
    conv_lengths_turns = []
    conv_lengths_tokens = []

    for ex in sft_examples:
        nrev = ex.get("num_revisions", 0)
        sft_rev_dist[nrev] += 1
        if nrev == 3:
            if ex.get("final_correct", False):
                sft_rev_split["3_solved"] += 1
            else:
                sft_rev_split["3_failed"] += 1
        else:
            sft_rev_split[str(nrev)] += 1

        if ex.get("final_correct", False):
            sft_final_correct += 1
        if ex.get("has_final_confirmation", False):
            sft_final_conf += 1

        for et in ex.get("error_types_addressed", []):
            sft_error_type_counts[et] += 1
            lab = str(et).lower()
            lab = {
                "interpretation": "Interpretation",
                "conceptual": "Conceptual",
                "structural": "Structural",
                "computational": "Computational",
            }.get(lab, "Other")
            sft_error_type_clean[lab] += 1

        conv_lengths_turns.append(len(ex.get("dialogue", [])))
        if tokenizer:
            text = " ".join([m.get("content", "") for m in ex.get("dialogue", [])])
            conv_lengths_tokens.append(tokenize_len(tokenizer, text))

    # 6) DPO stats
    rejection_reason_counts = Counter()
    prompt_token_lengths = []
    chosen_token_lengths = []
    rejected_token_lengths = []
    chosen_rev_dist = Counter()
    chosen_rev_split = Counter()

    for ex in dpo_pairs:
        meta = ex.get("metadata", {})
        revs = meta.get("chosen_revisions", -1)
        chosen_rev_dist[revs] += 1
        if revs == 3:
            chosen_rev_split["3_revs"] += 1
        else:
            chosen_rev_split[str(revs)] += 1

        rejection_reason_counts[meta.get("rejection_reason", "unknown")] += 1

        if tokenizer:
            prompt_token_lengths.append(
                tokenize_len(tokenizer, " ".join(m["content"] for m in ex.get("prompt", [])))
            )
            chosen_token_lengths.append(
                tokenize_len(tokenizer, " ".join(m["content"] for m in ex.get("chosen", [])))
            )
            rejected_token_lengths.append(
                tokenize_len(
                    tokenizer, " ".join(m["content"] for m in ex.get("rejected", []))
                )
            )

    # 7) Teacher feedback stats (correctly from teacher feedback lines)
    fb_lengths = []
    for r in results:
        fbs = extract_teacher_feedback_texts(r.get("dialogue", []))
        for fb in fbs:
            fb_lengths.append(len(fb.split()))
    fb_stats = compute_teacher_feedback_stats(results)

    # 8) Trace-quality breakdown
    df_quality, quality_totals = compute_trace_quality(results, sft_examples)
    df_quality.to_csv(TRACE_QUALITY_CSV, index=False)

    # 9) PLOTS
    plot_revision_distribution_split(
        sft_examples, os.path.join(PLOT_DIR, "revision_distribution_split.png")
    )
    plot_error_type_frequency(
        sft_examples, os.path.join(PLOT_DIR, "error_type_frequency_clean.png")
    )
    plot_error_type_revision_success_heatmap(
        results, os.path.join(PLOT_DIR, "error_type_revision_success_heatmap.png")
    )
    # Difficulty overlay (only if we have rates)
    if train_rates or fresh_rates or (old_test_rates and len(old_test_rates) > 0):
        plot_difficulty_overlay(
            train_rates,
            old_test_rates if old_test_rates is not None else None,
            fresh_rates,
            os.path.join(PLOT_DIR, "difficulty_hist_overlay.png"),
        )
    plot_trace_quality_breakdown(
        df_quality, os.path.join(PLOT_DIR, "trace_quality_breakdown.png")
    )
    plot_teacher_feedback_length_hist(
        fb_stats, fb_lengths, os.path.join(PLOT_DIR, "teacher_feedback_length_hist.png")
    )

    # 10) PRINT SUMMARY + SAVE JSON
    sft_total = len(sft_examples)
    dpo_total = len(dpo_pairs)
    summary_table = {
        # Big‑Math meta + dedup/filter
        "big_math_n_after_filter": len(df_big_math),
        **big_math_meta,
        # Train/Test difficulty coverage
        "train_examples": len(train_examples),
        "train_solve_rates_recovered": len(train_rates),
        "old_test_examples": len(old_test_examples),
        "old_test_solve_rates_recovered": 0
        if old_test_rates is None
        else len(old_test_rates),
        "fresh_test_examples": len(fresh_examples),
        # SFT stats
        "sft_total_examples": sft_total,
        "sft_final_correct_pct": round(
            (sft_final_correct / sft_total * 100.0), 2
        )
        if sft_total > 0
        else 0.0,
        "sft_final_confirmation_pct": round(
            (sft_final_conf / sft_total) * 100.0, 2
        )
        if sft_total > 0
        else 0.0,
        "sft_avg_turns": round(mean(conv_lengths_turns), 2)
        if conv_lengths_turns
        else 0.0,
        "sft_avg_tokens": round(mean(conv_lengths_tokens), 2)
        if conv_lengths_tokens
        else 0.0,
        # DPO stats
        "dpo_total_pairs": dpo_total,
        "dpo_avg_prompt_tokens": round(mean(prompt_token_lengths), 2)
        if prompt_token_lengths
        else 0.0,
        "dpo_avg_chosen_tokens": round(mean(chosen_token_lengths), 2)
        if chosen_token_lengths
        else 0.0,
        "dpo_avg_rejected_tokens": round(mean(rejected_token_lengths), 2)
        if rejected_token_lengths
        else 0.0,
        # Teacher feedback
        "teacher_feedback_count": fb_stats.get("count", 0),
        "teacher_feedback_avg_words": fb_stats.get("avg_words", 0.0),
        "teacher_feedback_median_words": fb_stats.get("median_words", 0.0),
        "teacher_feedback_p95_words": fb_stats.get("p95_words", 0.0),
        # Trace quality
        "trace_quality_totals": quality_totals,
    }

    # Also include trace‑quality breakdown in the JSON
    tq_rows = df_quality.to_dict(orient="records")
    summary_out = {"summary": summary_table, "trace_quality_rows": tq_rows}
    save_json(SUMMARY_JSON, summary_out)

    # Console print (concise)
    print("\n=== DATA GENERATION SUMMARY ===")
    for k, v in summary_table.items():
        print(f"{k}: {v}")

    print("\n=== TRACE QUALITY BREAKDOWN ===")
    print(df_quality)


if __name__ == "__main__":
    main()