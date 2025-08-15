import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"  # Use all 6 GPUs
from unsloth import PatchDPOTrainer
PatchDPOTrainer()
from unsloth import FastLanguageModel
import torch

max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# Path to your SFT LoRA adapter
SFT_LORA_PATH = "/home/sai/dc-distil/pipeline/lora_approach_2_20000_samples"  # Update this with your actual path

# Option 1: Load the SFT model directly and use it for DPO
# This will continue training the existing LoRA adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = SFT_LORA_PATH,  # Load your fine-tuned model directly
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map = "auto",  # This will distribute across GPUs
)

# For DPO, we need a reference model. We'll load the same SFT model separately
ref_model, _ = FastLanguageModel.from_pretrained(
    model_name = SFT_LORA_PATH,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map = "auto",
)

# Make sure the reference model is in eval mode and frozen
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# @title Alignment Handbook utils
import os
import re
from typing import List, Literal, Optional
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"] = "sft",
    assistant_prefix="<|assistant|>\n",
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            prompt_messages = [
                [msg for msg in example["chosen"] if msg["role"] == "user"][0]
            ]
            # Insert system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])
            # TODO: handle case where chosen/rejected also have system messages
            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_chosen"] = _strip_prefix(
                example["text_chosen"], assistant_prefix
            )
            example["text_rejected"] = _strip_prefix(
                example["text_rejected"], assistant_prefix
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example

def get_datasets(
    data_config: dict,
    splits: List[str] = ["train", "test"],
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.
    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """
    if type(data_config) is dict:
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")
    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    return raw_datasets

def mix_datasets(
    dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.
    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, split=split)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(
                    f"Split type {split} not recognized as one of test or train."
                )
    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")
    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(
                seed=42
            )
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)
    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )
    return raw_datasets

raw_datasets = get_datasets(
    {"saital/llama3_8b_self_revision_dpo_50K" : 0.4}, # 0.5% sampled
    splits = ["train"],
)
column_names = list(raw_datasets["train"].features)
raw_datasets = raw_datasets.map(
    apply_chat_template,
    fn_kwargs = {"tokenizer": tokenizer, "task": "dpo"},
    num_proc = 12,
    remove_columns = column_names,
    desc = "Formatting comparisons with prompt template",
)

# Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
for split in ["train"]:
    raw_datasets[split] = raw_datasets[split].rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    )

# Calculate dynamic max_length and max_prompt_length
def calculate_max_lengths(dataset, tokenizer, percentile=95):
    """
    Calculate appropriate max lengths based on the dataset
    """
    prompt_lengths = []
    total_lengths = []  # Combined list for both chosen and rejected
    
    # Sample a subset for efficiency
    sample_size = min(1000, len(dataset))
    indices = torch.randperm(len(dataset))[:sample_size].tolist()
    for idx in indices:
        example = dataset[idx]
        prompt_tokens = tokenizer(example["prompt"], return_tensors="pt").input_ids.shape[1]
        chosen_tokens = tokenizer(example["chosen"], return_tensors="pt").input_ids.shape[1]
        rejected_tokens = tokenizer(example["rejected"], return_tensors="pt").input_ids.shape[1]
        
        prompt_lengths.append(prompt_tokens)
        # Add both types of sequences to the same list
        total_lengths.append(prompt_tokens + chosen_tokens)
        total_lengths.append(prompt_tokens + rejected_tokens)
    
    # Calculate percentiles from the combined list
    max_prompt_length = int(torch.tensor(prompt_lengths).float().quantile(percentile/100))
    max_length = int(torch.tensor(total_lengths).float().quantile(percentile/100))
    
    # Round up to nearest 128 for efficiency
    max_prompt_length = ((max_prompt_length + 127) // 128) * 128
    max_length = ((max_length + 127) // 128) * 128
    
    # Cap at model's max sequence length
    max_length = min(max_length, max_seq_length)
    max_prompt_length = min(max_prompt_length, max_length // 2)  # Ensure prompt is at most half
    
    print(f"Calculated max_prompt_length: {max_prompt_length}")
    print(f"Calculated max_length: {max_length}")
    return max_length, max_prompt_length

# Calculate dynamic lengths
max_length, max_prompt_length = calculate_max_lengths(raw_datasets["train"], tokenizer)

from unsloth import PatchDPOTrainer
PatchDPOTrainer()

from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig

# Adjust batch size based on number of GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
per_device_train_batch_size = 16  # You can increase this if you have memory headroom
gradient_accumulation_steps = max(1, 8 // num_gpus)  # Adjust to maintain effective batch size

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = ref_model,  # Use the separately loaded reference model
    args = DPOConfig(
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 5e-6,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        # Multi-GPU settings
        ddp_find_unused_parameters = False,
        gradient_checkpointing = True,
        # Save settings
        save_strategy = "epoch",
        save_total_limit = 2,
        # Optional: Add deepspeed config for even better multi-GPU efficiency
        # deepspeed = "path/to/deepspeed_config.json",
    ),
    beta = 0.1,
    train_dataset = raw_datasets["train"],
    # eval_dataset = raw_datasets["test"],
    tokenizer = tokenizer,
    max_length = max_length,
    max_prompt_length = max_prompt_length,
)

# Train the model
dpo_trainer.train()

# Save the DPO-enhanced model
model.save_pretrained("sft_plus_dpo_lora_10")
tokenizer.save_pretrained("sft_plus_dpo_lora_10")