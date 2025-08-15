# train_sft_dp.py
import os
import torch
import sys

# Force CUDA initialization before importing Unsloth
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

# Force CUDA initialization
if torch.cuda.is_available():
    torch.cuda.init()
    # Set device based on local rank if available
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"Process {os.getpid()} using GPU {local_rank}")
    else:
        print(f"Process {os.getpid()} initialized CUDA")
else:
    print(f"Process {os.getpid()} - CUDA NOT AVAILABLE!")
    sys.exit(1)

# Now import Unsloth - should detect GPUs properly
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    print(f"Process {os.getpid()} successfully imported Unsloth")
except Exception as e:
    print(f"Process {os.getpid()} failed to import Unsloth: {str(e)}")
    sys.exit(1)

# Rest of imports
import json
from datasets import load_dataset, Dataset
import pandas as pd
import random
from typing import Literal
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
import shutil

# Initialize accelerator
accelerator = Accelerator()

# Add argument parser for approach selection
parser = argparse.ArgumentParser(description='Train self-revision model with different sampling approaches')
parser.add_argument('--approach', type=int, choices=[1, 2], default=1,
                   help='Training approach: 1 for mixed training, 2 for curriculum-based training')
parser.add_argument('--num_samples', type=int, default=1000,
                   help='Total number of samples to use (will be scaled according to percentages)')
parser.add_argument('--seed', type=int, default=3407,
                   help='Random seed for reproducibility')
args = parser.parse_args()

# Set random seed
set_seed(args.seed)

# Load model and tokenizer using Unsloth
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
max_seq_length = 14000

# Only print on main process
if accelerator.is_main_process:
    print("Loading model and tokenizer...")
    print(f"Device: {accelerator.device}")
    print(f"Number of processes: {accelerator.num_processes}")
    print(f"Process index: {accelerator.process_index}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# FIX: Use device_map to load directly on the correct device
# For 4-bit/8-bit models, we can't use .to() later, so load directly on the right device
device_map = {"": accelerator.process_index} if torch.cuda.is_available() else None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
    device_map=device_map,  # Load directly on the correct device
)

# Remove this line - it causes the error for 4-bit models
# model = model.to(accelerator.device)

# Add LoRA adapters
if accelerator.is_main_process:
    print("Adding LoRA adapters...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=args.seed,
)

# Dataset processing on main process only
temp_dataset_path = f"temp_dataset_{args.approach}_{args.num_samples}.arrow"

if accelerator.is_main_process:
    print("Loading dataset...")
    dataset = load_dataset("saital/llama3_8b_self_revision_sft_50K", split="train")
    
    # Check the structure of the first example
    print("\nChecking dataset structure...")
    print(f"First example keys: {list(dataset[0].keys())}")
    print(f"Type of dialogue field: {type(dataset[0]['dialogue'])}")

    # Parse JSON fields only if needed
    def parse_dialogue(example):
        # Check if dialogue is already a list or needs parsing
        if isinstance(example['dialogue'], str):
            example['dialogue'] = json.loads(example['dialogue'])
        # If it's already a list, we don't need to do anything
        return example

    # Only apply parsing if needed
    if isinstance(dataset[0]['dialogue'], str):
        print("Parsing JSON strings...")
        dataset = dataset.map(parse_dialogue)
    else:
        print("Dialogue field is already parsed")

    # Verify dialogue structure for self-revision training
    def verify_dialogue_structure(example):
        """Verify that dialogues follow the expected self-revision pattern"""
        dialogue = example['dialogue']
        
        # Check for [NO_REV] flag in the last assistant message
        if dialogue and dialogue[-1]['role'] == 'assistant':
            last_message = dialogue[-1]['content']
            has_no_rev = '[NO_REV]' in last_message
            example['has_no_rev_flag'] = has_no_rev
        else:
            example['has_no_rev_flag'] = False
        
        return example

    # Verify all dialogues
    dataset = dataset.map(verify_dialogue_structure)

    # Print statistics about [NO_REV] flags
    no_rev_count = sum(1 for ex in dataset if ex.get('has_no_rev_flag', False))
    print(f"\nDialogues with [NO_REV] flag: {no_rev_count}/{len(dataset)} ({no_rev_count/len(dataset)*100:.1f}%)")

    # Function to sample data according to revision distribution
    def sample_by_revisions(dataset, revision_distribution, total_samples):
        """
        Sample data according to revision distribution.
        revision_distribution: dict mapping num_revisions to percentage (0.0-1.0)
        total_samples: total number of samples to select
        """
        # Group dataset by number of revisions
        revision_groups = {}
        for i, example in enumerate(dataset):
            num_rev = example['num_revisions']
            if num_rev not in revision_groups:
                revision_groups[num_rev] = []
            revision_groups[num_rev].append(i)
        
        # Print available samples per revision count
        print("\nAvailable samples by revision count:")
        for num_rev in sorted(revision_groups.keys()):
            print(f"  {num_rev} revisions: {len(revision_groups[num_rev])} samples")
        
        # Sample according to distribution
        sampled_indices = []
        for num_rev, percentage in revision_distribution.items():
            target_count = int(total_samples * percentage)
            if num_rev in revision_groups:
                available_indices = revision_groups[num_rev]
                if len(available_indices) >= target_count:
                    sampled = random.sample(available_indices, target_count)
                else:
                    # If not enough samples, take all available and log warning
                    sampled = available_indices
                    print(f"Warning: Only {len(available_indices)} samples available for {num_rev} revisions, "
                          f"but {target_count} requested")
                sampled_indices.extend(sampled)
            else:
                print(f"Warning: No samples found with {num_rev} revisions")
        
        return sampled_indices

    # Define sampling approaches
    if args.approach == 1:
        print(f"\nUsing Approach 1: Mixed training with {args.num_samples} samples")
        
        # Approach 1: Mixed training
        revision_distribution = {
            0: 0.30,  # 30%
            1: 0.25,  # 25%
            2: 0.25,  # 25%
            3: 0.20   # 20% (remaining)
        }
        
        sampled_indices = sample_by_revisions(dataset, revision_distribution, args.num_samples)
        random.shuffle(sampled_indices)  # Shuffle for mixed training
        final_dataset = dataset.select(sampled_indices)
        
        print(f"Final dataset size: {len(final_dataset)} samples")
        
        # Print distribution
        rev_counts = {}
        for example in final_dataset:
            num_rev = example['num_revisions']
            rev_counts[num_rev] = rev_counts.get(num_rev, 0) + 1
        print("\nActual distribution:")
        for num_rev in sorted(rev_counts.keys()):
            print(f"  {num_rev} revisions: {rev_counts[num_rev]} samples ({rev_counts[num_rev]/len(final_dataset)*100:.1f}%)")

    else:  # args.approach == 2
        print(f"\nUsing Approach 2: Curriculum-based training with {args.num_samples} samples total")
        
        # Calculate samples for each phase based on total
        phase_samples = {
            1: int(args.num_samples * 0.30),  # 30% for phase 1
            2: int(args.num_samples * 0.30),  # 30% for phase 2
            3: int(args.num_samples * 0.25),  # 25% for phase 3
            4: int(args.num_samples * 0.15),  # 15% for phase 4
        }
        
        # Adjust for rounding errors
        total_allocated = sum(phase_samples.values())
        if total_allocated < args.num_samples:
            phase_samples[1] += args.num_samples - total_allocated
        
        # Phase distributions
        phase_distributions = {
            1: {0: 0.50, 1: 0.50},
            2: {0: 0.20, 1: 0.40, 2: 0.40},
            3: {0: 0.10, 1: 0.20, 2: 0.30, 3: 0.40},
            4: {0: 0.35, 1: 0.25, 2: 0.25, 3: 0.15}
        }
        
        # Sample for each phase
        all_sampled_indices = []
        for phase, num_samples in phase_samples.items():
            print(f"\nPhase {phase}: {num_samples} samples")
            distribution = phase_distributions[phase]
            
            phase_indices = sample_by_revisions(dataset, distribution, num_samples)
            all_sampled_indices.extend(phase_indices)
            
            # Print phase distribution
            phase_dataset = dataset.select(phase_indices)
            rev_counts = {}
            for example in phase_dataset:
                num_rev = example['num_revisions']
                rev_counts[num_rev] = rev_counts.get(num_rev, 0) + 1
            print(f"  Phase {phase} distribution:")
            for num_rev in sorted(rev_counts.keys()):
                print(f"    {num_rev} revisions: {rev_counts[num_rev]} samples ({rev_counts[num_rev]/len(phase_dataset)*100:.1f}%)")
        
        # Create final dataset maintaining phase order (curriculum)
        final_dataset = dataset.select(all_sampled_indices)
        print(f"\nTotal dataset size: {len(final_dataset)} samples")

    # Apply chat template to each COMPLETE conversation
    print("\nApplying chat template to complete conversations...")
    formatted_conversations = []

    for idx, example in enumerate(final_dataset):
        dialogue = example["dialogue"]
        
        # Verify this is a complete conversation
        if len(dialogue) < 3:  # At minimum: system, user, assistant
            print(f"Warning: Conversation {idx} has only {len(dialogue)} messages")
            continue
        
        # Count revision rounds (user prompts asking for revision)
        revision_prompts = sum(1 for msg in dialogue if msg['role'] == 'user' and 
                              'analyze your previous solution' in msg.get('content', '').lower())
        
        # Apply chat template to the ENTIRE conversation as one unit
        formatted = tokenizer.apply_chat_template(
            dialogue,  # This is the complete conversation
            tokenize=False,
            add_generation_prompt=False  # No generation prompt for training
        )
        
        formatted_conversations.append(formatted)
        
        # Debug: print info about first few conversations
        if idx < 3:
            print(f"\nExample {idx}:")
            print(f"  Total messages: {len(dialogue)}")
            print(f"  Revision prompts: {revision_prompts}")
            print(f"  Has [NO_REV]: {example.get('has_no_rev_flag', False)}")
            print(f"  Formatted length: {len(formatted)} characters")

    # Create dataset from formatted conversations
    print(f"\nCreated {len(formatted_conversations)} formatted conversations")

    # Verify a complete conversation is preserved
    print("\n=== VERIFICATION: Complete conversation example ===")
    print("First 2000 characters of formatted conversation:")
    print(formatted_conversations[0][:2000])
    print("\n... (conversation continues)")
    print("\nLast 500 characters (should contain [NO_REV]):")
    print(formatted_conversations[0][-500:])

    # Create a pandas DataFrame and then convert to Dataset
    data = pd.DataFrame({'text': formatted_conversations})
    formatted_dataset = Dataset.from_pandas(data)
    
    # Save the formatted dataset to disk
    formatted_dataset.save_to_disk(temp_dataset_path)
    print(f"\nSaved formatted dataset to {temp_dataset_path}")

# Synchronize all processes
accelerator.wait_for_everyone()

# All processes load the dataset from disk
formatted_dataset = Dataset.load_from_disk(temp_dataset_path)

if accelerator.is_main_process:
    print(f"\nAll processes loaded dataset with {len(formatted_dataset)} examples")

from trl import SFTTrainer, SFTConfig

# Adjust training parameters based on approach
if args.approach == 1:
    num_epochs = 3
    warmup_steps = 5
else:  # Curriculum approach might benefit from fewer epochs since data is ordered
    num_epochs = 4
    warmup_steps = 10

# Adjust batch size for multi-GPU
# With 6 GPUs, we can use a larger effective batch size
per_device_batch_size = 3  # Per GPU
gradient_accumulation_steps = 4  # Less accumulation needed with more GPUs
effective_batch_size = per_device_batch_size * gradient_accumulation_steps * accelerator.num_processes

if accelerator.is_main_process:
    print("\n=== IMPORTANT: Training Configuration ===")
    print("Each training example is a COMPLETE multi-turn conversation including:")
    print("1. Initial problem solving")
    print("2. Multiple self-revision prompts and responses")
    print("3. Final response with [NO_REV] flag")
    print("This trains the model to handle the full self-revision loop.")
    print(f"\nMulti-GPU Configuration:")
    print(f"  Number of GPUs: {accelerator.num_processes}")
    print(f"  Per GPU batch size: {per_device_batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print("==========================================\n")

# Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=SFTConfig(
        dataset_text_field="text",  # Each 'text' is a complete conversation
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        learning_rate=5e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=f"outputs_approach_{args.approach}",
        report_to="none",
        
        # Important for long conversations
        max_seq_length=max_seq_length,
        
        # Packing can be useful for efficient training with variable length conversations
        packing=False,  # Set to True if you want to pack multiple conversations in one sequence
        
        # DDP settings
        ddp_find_unused_parameters=False,
        
        # Important: set to 0 to avoid issues
        dataloader_num_workers=0,
        
        # Save strategy
        save_strategy="no",  # We'll save manually at the end
    ),
)

if accelerator.is_main_process:
    print(f"\nStarting training with approach {args.approach}...")
    print(f"Total samples: {len(formatted_dataset)}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Total training steps: {len(formatted_dataset) * num_epochs // effective_batch_size}")

trainer.train()

# Save model only on main process
if accelerator.is_main_process:
    print("\nSaving model...")
    save_path = f"lora_approach_{args.approach}_{args.num_samples}_samples"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save training configuration with self-revision details
    config_info = {
        "approach": args.approach,
        "num_samples": args.num_samples,
        "actual_samples": len(formatted_dataset),
        "seed": args.seed,
        "num_epochs": num_epochs,
        "num_gpus": accelerator.num_processes,
        "effective_batch_size": effective_batch_size,
        "approach_description": "Mixed training" if args.approach == 1 else "Curriculum-based training",
        "training_objective": "Self-revision with [NO_REV] termination",
        "expected_inference_behavior": "User repeatedly prompts with self-revision request until model responds with [NO_REV]"
    }
    
    with open(f"{save_path}/training_config.json", "w") as f:
        json.dump(config_info, f, indent=2)
    
    # Clean up temporary dataset file
    if os.path.exists(temp_dataset_path):
        shutil.rmtree(temp_dataset_path)
        print(f"Cleaned up temporary dataset file")
    
    print(f"\nTraining complete! Model saved to {save_path}/")
    print(f"Training approach: {config_info['approach_description']}")
    print(f"Total samples used: {len(formatted_dataset)}")
    print("\n=== Inference Usage ===")
    print("During inference, repeatedly prompt with:")
    print('"Please analyze your previous solution thoroughly..."')
    print("Until the model responds with [NO_REV] flag")

# Wait for all processes to finish
accelerator.wait_for_everyone()

if accelerator.is_main_process:
    print("\nAll processes completed successfully!")