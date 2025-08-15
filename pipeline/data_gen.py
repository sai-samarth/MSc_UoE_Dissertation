# diffusion_revision_trace_generator_multi_gpu.py
import json
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging
import os
import time
from pathlib import Path
import random
from enum import Enum
from collections import defaultdict
import re
from datasets import load_dataset
import pandas as pd
from vllm import LLM, SamplingParams
import torch
import multiprocessing as mp
from multiprocessing import Queue, Process
import queue
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProblemStage(Enum):
    INITIAL_GENERATION = 0
    INITIAL_CHECK = 1
    FEEDBACK_1 = 2
    REVISION_1 = 3
    CHECK_1 = 4
    FEEDBACK_2 = 5
    REVISION_2 = 6
    CHECK_2 = 7
    FEEDBACK_3 = 8
    REVISION_3 = 9
    CHECK_3 = 10
    FINAL_CONFIRMATION = 11  # For adding final confirmation
    COMPLETED = 12

@dataclass
class Config:
    # Model configurations
    teacher_model_name: str = "Qwen/Qwen3-8B-AWQ"
    student_model_name: str = "lurker18/Llama_3.1_8B_Instruct_AWQ_4bit"
    
    # Generation parameters
    teacher_temperature: float = 0.2
    student_temperature: float = 0.7
    max_tokens: int = 2048
    max_revisions: int = 3
    max_revision_attempts: int = 3
    max_boxed_answer_retries: int = 3
    
    # Dataset parameters
    dataset_name: str = "SynthLabsAI/Big-Math-RL-Verified"
    num_samples: int = 200000
    train_ratio: float = 0.8
    min_solve_rate: float = 0.2
    max_solve_rate: float = 0.7
    
    # Output parameters
    output_dir: str = "diffusion_revision_traces"
    batch_size: int = 2500  # Per GPU batch size
    
    # Multi-GPU parameters
    num_gpus: int = 6
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: int = 10000
    teacher_max_model_len: int = 8000
    
    # Resume parameters
    checkpoint_interval: int = 100

SYSTEM_PROMPT_STUDENT = """You are a student working on mathematical problems. Show your step-by-step reasoning and provide the final answer within \\boxed{}."""

SYSTEM_PROMPT_TEACHER = """You are a helpful math teacher providing targeted feedback. Be encouraging, constructive, and guide without revealing answers. Use Socratic questions to prompt rethinking."""

FINAL_CONFIRMATION_GENERATION_PROMPT = """Your previous solution appears to be correct based on your analysis. Please perform a final self-review as if you're double-checking your work independently. Provide a brief (2 sentence) reflection confirming your approach and answer look correct, then add [NO_REV] at the end to indicate no revision is needed."""

# Standardized self-revision prompt that will be used during finetuning
STANDARDIZED_SELF_REVISION_PROMPT = """Please analyze your previous solution thoroughly and identify any potential errors or areas for improvement. Carefully check your:
- Problem interpretation and understanding
- Mathematical reasoning and logic
- Calculations and arithmetic
- Final answer format and completeness

If you find any mistakes, first explain in 2 sentences: (1) what specific error you made, (2) how you'll correct it. Then provide your revised solution.

If you're confident your solution is correct after careful review, briefly explain why (2 sentences) and confirm no revision is needed by adding [NO_REV] at the end."""

@dataclass
class RevisionAttempt:
    """Store individual revision attempts for DPO pairs"""
    revision_text: str
    addressed_feedback: bool
    revision_stage: int
    error_type: str

class DiffusionRevisionSystem:
    """Implements flexible revision with critical error identification"""
    
    @staticmethod
    def analyze_solution(problem: str, solution: str, expected_answer: str) -> str:
        """Identify the most critical error in the solution"""
        
        return f"""Analyze this student solution and identify the MOST CRITICAL error that needs to be fixed first.

Problem: {problem}

Student Solution:
{solution}

Expected Answer (for your reference only, DO NOT reveal): {expected_answer}

Identify the single most important issue that's preventing the correct solution. This could be:
- A conceptual misunderstanding (wrong approach/method)
- A structural error (missing steps, wrong formula application)
- A computational mistake (arithmetic errors)
- An interpretation error (misreading the problem)

Provide ONE specific guidance that addresses this critical error without revealing the answer.

Format:
ERROR_TYPE: [conceptual/structural/computational/interpretation]
FEEDBACK: [One specific Socratic question or hint targeting the critical error]"""
    
    @staticmethod
    def create_revision_prompt(feedback: str, error_type: str) -> str:
        """Create a prompt that encourages natural understanding and revision based on feedback"""
        
        return f"""Teacher feedback: {feedback}

Now, take a step back and think through your solution again. Without repeating what the teacher said, reflect on your work as if you're discovering the issue yourself. Write 2 sentences in a natural way:
- What specific mistake or oversight did you just realize you made?
- What was your original thinking that led to this error?
- What's the correct approach you should take instead?

Write as if you're having an "aha!" moment - like you just noticed something you missed before. 

Then, using your new understanding from this reflection, provide your complete revised solution that directly addresses the issue you identified. Show all steps clearly and put your final answer in \\boxed{{}}.

Remember: Your reflection should sound like your own realization, not a response to feedback."""

    @staticmethod
    def check_revision_follows_feedback(original_solution: str, revised_solution: str, feedback: str) -> str:
        """Check if the student's revision actually addresses the feedback"""
        
        return f"""Evaluate if the student's revision adequately addresses the teacher's feedback.

Original Solution:
{original_solution}

Teacher Feedback:
{feedback}

Student's Revised Solution:
{revised_solution}

Did the student:
1. Acknowledge the issue raised in the feedback?
2. Addressed the feeback in the revised answer?
3. Show understanding of what needed to be fixed?

Answer with just:
ADDRESSED: [YES if the revision addresses the feedback, NO if it ignores or misunderstands it]"""

def extract_thinking_content(text: str) -> str:
    """Remove thinking tags and extract clean content"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()
    return text

def extract_feedback_components(response: str) -> Tuple[str, str]:
    """Extract error type and feedback from teacher response"""
    try:
        clean_response = extract_thinking_content(response)
        
        error_match = re.search(r'ERROR_TYPE:\s*(\w+)', clean_response, re.IGNORECASE)
        feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?=$)', clean_response, re.DOTALL)
        
        error_type = error_match.group(1).lower() if error_match else "general"
        feedback = feedback_match.group(1).strip() if feedback_match else "Please review your solution carefully."
        
        return error_type, feedback
    except Exception as e:
        logger.error(f"Error parsing feedback: {e}")
        return "general", "Please review your solution carefully."

def check_revision_addressed(response: str) -> bool:
    """Check if revision addresses feedback"""
    try:
        clean_response = extract_thinking_content(response)
        
        addressed_match = re.search(r'ADDRESSED:\s*(\w+)', clean_response, re.IGNORECASE)
        addressed = addressed_match and "YES" in addressed_match.group(1).upper()
        return addressed
    except Exception as e:
        logger.error(f"Error parsing revision check: {e}")
        return False

def has_boxed_answer(text: str) -> bool:
    """Check if text contains a boxed answer"""
    boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    return bool(re.search(boxed_pattern, text))

def has_no_revision_flag(text: str) -> bool:
    """Check if text contains [NO_REV] flag"""
    return "[NO_REV]" in text

def model_worker(gpu_id: int, config: Config, input_queue: Queue, output_queue: Queue):
    """Worker process that can load either student or teacher model dynamically"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    logger.info(f"Starting worker on GPU {gpu_id}")
    
    current_model = None
    current_model_type = None
    llm = None
    
    while True:
        try:
            batch_data = input_queue.get(timeout=30)
            
            if batch_data is None:  # Shutdown signal
                logger.info(f"Worker on GPU {gpu_id} shutting down")
                break
            
            model_type, task_type, batch_id, prompts = batch_data
            
            # Load model if needed
            if current_model_type != model_type:
                # Unload current model
                if llm is not None:
                    del llm
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(f"Unloaded {current_model_type} model from GPU {gpu_id}")
                
                # Load new model
                if model_type == "student":
                    logger.info(f"Loading student model on GPU {gpu_id}")
                    llm = LLM(
                        model=config.student_model_name,
                        # quantization="awq",
                        # dtype="half",
                        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
                        max_model_len=config.vllm_max_model_len,
                        trust_remote_code=True,
                        # tensor_parallel_size=1
                    )
                    current_model_type = "student"
                else:  # teacher
                    logger.info(f"Loading teacher model on GPU {gpu_id}")
                    llm = LLM(
                        model=config.teacher_model_name,
                        # quantization="awq",
                        # dtype="half",
                        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
                        max_model_len=config.teacher_max_model_len,
                        trust_remote_code=True,
                        # tensor_parallel_size=1
                    )
                    current_model_type = "teacher"
            
            # Set sampling params based on model and task
            if model_type == "student":
                sampling_params = SamplingParams(
                    temperature=config.student_temperature,
                    max_tokens=config.max_tokens,
                    stop=["<|eot_id|>", "<|end_of_text|>"]
                )
            else:  # teacher
                if task_type == "feedback":
                    sampling_params = SamplingParams(
                        temperature=config.teacher_temperature,
                        max_tokens=4096,
                        stop=["<|eot_id|>", "<|end_of_text|>"]
                    )
                else:  # check tasks
                    sampling_params = SamplingParams(
                        temperature=0.0,
                        max_tokens=4096,
                        stop=["<|eot_id|>", "<|end_of_text|>"]
                    )
            
            # Generate responses
            outputs = llm.generate(prompts, sampling_params)
            responses = []
            
            for output in outputs:
                raw_response = output.outputs[0].text.strip()
                if model_type == "teacher":
                    # Extract content without thinking tags for teacher
                    clean_response = extract_thinking_content(raw_response)
                    responses.append(clean_response)
                else:
                    responses.append(raw_response)
            
            # Send results back
            output_queue.put((task_type, batch_id, responses))
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in worker on GPU {gpu_id}: {e}")
            if 'batch_id' in locals():
                output_queue.put((task_type, batch_id, ["Error"] * len(prompts)))

class ProblemState:
    def __init__(self, problem_data: Dict):
        self.problem_id = problem_data["problem_id"]
        self.problem = problem_data["problem"]
        self.expected_answer = problem_data["expected_answer"]
        self.stage = ProblemStage.INITIAL_GENERATION
        self.revisions = 0
        self.revision_attempts = 0
        self.solved = False
        self.current_solution = None
        self.previous_solution = None
        self.dialogue = []
        self.pending_feedback = None
        self.pending_error_type = None
        self.error_types_addressed = []
        self.revision_attempts_history = []
        self.failed_due_to_poor_revisions = False
        self.boxed_answer_retries = 0
        self.final_confirmation_added = False

class DynamicModelManager:
    """Manages dynamic loading/unloading of models across all GPUs"""
    def __init__(self, config: Config):
        self.config = config
        self.input_queues = []
        self.output_queue = mp.Queue(maxsize=config.num_gpus * 10)
        self.workers = []
        self.request_counter = 0
        
        # Start worker processes on all GPUs
        for gpu_id in range(config.num_gpus):
            input_queue = mp.Queue(maxsize=10)
            self.input_queues.append(input_queue)
            
            worker = Process(
                target=model_worker,
                args=(gpu_id, config, input_queue, self.output_queue)
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {config.num_gpus} dynamic workers on all GPUs")
    
    def format_student_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for student model (Llama)"""
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            elif msg["role"] == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            elif msg["role"] == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt
    
    def format_teacher_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for teacher model (Qwen)"""
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    def extract_boxed_answer(self, solution: str) -> Optional[str]:
        """Extract answer from \\boxed{} format"""
        boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(boxed_pattern, solution)
        
        if matches:
            return matches[-1].strip()
        return None
    
    async def generate_student_batch(self, prompts: List[str], task_type: str = "generation") -> List[str]:
        """Generate responses using student model on all GPUs"""
        if not prompts:
            return []
        
        # Distribute prompts across all GPUs
        gpu_batches = [[] for _ in range(self.config.num_gpus)]
        for i, prompt in enumerate(prompts):
            gpu_idx = i % self.config.num_gpus
            gpu_batches[gpu_idx].append(prompt)
        
        # Send to workers
        batch_ids = []
        for gpu_idx, gpu_prompts in enumerate(gpu_batches):
            if gpu_prompts:
                batch_id = f"student_{self.request_counter}_{gpu_idx}"
                self.request_counter += 1
                batch_ids.append((batch_id, gpu_idx, len(gpu_prompts)))
                self.input_queues[gpu_idx].put(("student", task_type, batch_id, gpu_prompts))
        
        # Collect results
        results = [None] * len(prompts)
        collected = 0
        
        while collected < len(batch_ids):
            task_type_resp, batch_id, responses = await asyncio.get_event_loop().run_in_executor(
                None, self.output_queue.get
            )
            
            # Find which batch this is and map back
            for bid, gpu_idx, batch_len in batch_ids:
                if bid == batch_id:
                    # Map responses back to original positions
                    original_indices = [i for i in range(len(prompts)) if i % self.config.num_gpus == gpu_idx]
                    for i, response in enumerate(responses):
                        if i < len(original_indices):
                            results[original_indices[i]] = response
                    collected += 1
                    break
        
        return results
    
    async def generate_teacher_batch(self, requests: List[Tuple], task_type: str) -> Dict:
        """Generate responses using teacher model on all GPUs"""
        if not requests:
            return {}
        
        # Prepare prompts based on task type
        prompts = []
        problem_ids = []
        
        if task_type == "feedback":
            for problem_id, problem, solution, expected_answer in requests:
                feedback_prompt = DiffusionRevisionSystem.analyze_solution(
                    problem, solution, expected_answer
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_TEACHER},
                    {"role": "user", "content": feedback_prompt}
                ]
                prompt = self.format_teacher_prompt(messages)
                prompts.append(prompt)
                problem_ids.append(problem_id)
        
        elif task_type == "check_revision":
            for problem_id, original, revised, feedback in requests:
                check_prompt = DiffusionRevisionSystem.check_revision_follows_feedback(
                    original, revised, feedback
                )
                messages = [{"role": "user", "content": check_prompt}]
                prompt = self.format_teacher_prompt(messages)
                prompts.append(prompt)
                problem_ids.append(problem_id)
        
        elif task_type == "check_answer":
            valid_indices = []
            for i, (problem_id, solution, expected_answer) in enumerate(requests):
                student_answer = self.extract_boxed_answer(solution)
                
                if student_answer is None:
                    logger.info(f"Problem {problem_id}: No \\boxed{{}} answer found, marking as incorrect")
                    continue
                
                valid_indices.append(i)
                check_prompt = f"""Compare these two mathematical answers for exact equivalence:

Student's answer: {student_answer}
Expected answer: {expected_answer}

Are they mathematically equivalent? Consider:
- Simplified forms (e.g., 6/3 = 2, 4/2 = 2)
- Different representations of the same value (e.g., 0.5 = 1/2 = \\frac{{1}}{{2}})
- Algebraic equivalence (e.g., 2\\sqrt{{3}} = \\sqrt{{12}})
- Be very strict - they must represent the EXACT same value

Provide a very short (1 line) explanation of why they are or aren't equivalent, then put your final answer in \\boxed{{YES}} or \\boxed{{NO}}."""
                
                messages = [{"role": "user", "content": check_prompt}]
                prompt = self.format_teacher_prompt(messages)
                prompts.append(prompt)
                problem_ids.append(problem_id)
        
        if not prompts:
            # Return all False for check_answer if no valid prompts
            if task_type == "check_answer":
                return {req[0]: False for req in requests}
            return {}
        
        # Distribute prompts across all GPUs
        gpu_batches = [[] for _ in range(self.config.num_gpus)]
        gpu_problem_ids = [[] for _ in range(self.config.num_gpus)]
        
        for i, (prompt, problem_id) in enumerate(zip(prompts, problem_ids)):
            gpu_idx = i % self.config.num_gpus
            gpu_batches[gpu_idx].append(prompt)
            gpu_problem_ids[gpu_idx].append(problem_id)
        
        # Send to workers
        batch_ids = []
        for gpu_idx, gpu_prompts in enumerate(gpu_batches):
            if gpu_prompts:
                batch_id = f"teacher_{task_type}_{self.request_counter}_{gpu_idx}"
                self.request_counter += 1
                batch_ids.append((batch_id, gpu_idx, len(gpu_prompts)))
                self.input_queues[gpu_idx].put(("teacher", task_type, batch_id, gpu_prompts))
        
        # Collect results
        results = {}
        collected = 0
        
        while collected < len(batch_ids):
            task_type_resp, batch_id, responses = await asyncio.get_event_loop().run_in_executor(
                None, self.output_queue.get
            )
            
            # Find which batch this is
            for bid, gpu_idx, batch_len in batch_ids:
                if bid == batch_id:
                    # Map responses back to problem IDs
                    for i, response in enumerate(responses):
                        problem_id = gpu_problem_ids[gpu_idx][i]
                        
                        if task_type == "feedback":
                            results[problem_id] = response
                        elif task_type == "check_revision":
                            results[problem_id] = check_revision_addressed(response)
                        elif task_type == "check_answer":
                            response_boxed = self.extract_boxed_answer(response)
                            verdict = response_boxed and "YES" in response_boxed.upper()
                            results[problem_id] = verdict
                            logger.info(f"Problem {problem_id}: Verdict={verdict}")
                    
                    collected += 1
                    break
        
        # For check_answer, mark missing problems as False
        if task_type == "check_answer":
            for problem_id, _, _ in requests:
                if problem_id not in results:
                    results[problem_id] = False
        
        return results
    
    def shutdown(self):
        """Shutdown all workers"""
        logger.info("Shutting down workers...")
        for queue in self.input_queues:
            queue.put(None)
        
        for worker in self.workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()
        
        logger.info("All workers shut down")

def load_completed_problems(output_dir: str) -> Set[str]:
    """Load set of already completed problem IDs for resuming"""
    completed = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.json') and not filename.startswith('generation_'):
                problem_id = filename[:-5]  # Remove .json extension
                completed.add(problem_id)
    logger.info(f"Found {len(completed)} already completed problems")
    return completed

def save_checkpoint(problem_states: Dict[str, ProblemState], checkpoint_path: str):
    """Save current state of all problems for recovery"""
    checkpoint_data = {}
    for pid, state in problem_states.items():
        checkpoint_data[pid] = {
            "stage": state.stage.name,
            "revisions": state.revisions,
            "revision_attempts": state.revision_attempts,
            "solved": state.solved,
            "current_solution": state.current_solution,
            "dialogue": state.dialogue,
            "error_types_addressed": state.error_types_addressed,
            "revision_attempts_history": [
                {
                    "revision_text": attempt.revision_text,
                    "addressed_feedback": attempt.addressed_feedback,
                    "revision_stage": attempt.revision_stage,
                    "error_type": attempt.error_type
                }
                for attempt in state.revision_attempts_history
            ],
            "failed_due_to_poor_revisions": state.failed_due_to_poor_revisions,
            "boxed_answer_retries": state.boxed_answer_retries,
            "final_confirmation_added": state.final_confirmation_added
        }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)
    logger.debug(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path: str, problems: List[Dict]) -> Dict[str, ProblemState]:
    """Load problem states from checkpoint"""
    if not os.path.exists(checkpoint_path):
        return None
    
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)
    
    problem_states = {}
    problem_map = {p["problem_id"]: p for p in problems}
    
    for pid, state_data in checkpoint_data.items():
        if pid in problem_map:
            state = ProblemState(problem_map[pid])
            state.stage = ProblemStage[state_data["stage"]]
            state.revisions = state_data["revisions"]
            state.revision_attempts = state_data["revision_attempts"]
            state.solved = state_data["solved"]
            state.current_solution = state_data["current_solution"]
            state.dialogue = state_data["dialogue"]
            state.error_types_addressed = state_data["error_types_addressed"]
            state.revision_attempts_history = [
                RevisionAttempt(
                    revision_text=attempt["revision_text"],
                    addressed_feedback=attempt["addressed_feedback"],
                    revision_stage=attempt["revision_stage"],
                    error_type=attempt["error_type"]
                )
                for attempt in state_data.get("revision_attempts_history", [])
            ]
            state.failed_due_to_poor_revisions = state_data.get("failed_due_to_poor_revisions", False)
            state.boxed_answer_retries = state_data.get("boxed_answer_retries", 0)
            state.final_confirmation_added = state_data.get("final_confirmation_added", False)
            problem_states[pid] = state
    
    logger.info(f"Loaded checkpoint with {len(problem_states)} problems")
    return problem_states

def convert_dialogue_for_finetuning(dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert dialogue to use standardized self-revision prompts"""
    finetuning_dialogue = []
    
    for i, message in enumerate(dialogue):
        if message["role"] == "user" and "Teacher feedback:" in message["content"]:
            # Replace specific teacher feedback with standardized self-revision prompt
            finetuning_dialogue.append({
                "role": "user",
                "content": STANDARDIZED_SELF_REVISION_PROMPT
            })
        else:
            finetuning_dialogue.append(message)
    
    return finetuning_dialogue

async def process_problems_pipeline(
    problems: List[Dict], 
    model_manager: DynamicModelManager,
    config: Config,
    output_dir: str
) -> List[Dict]:
    
    # Check for existing completed problems
    completed_problem_ids = load_completed_problems(output_dir)
    
    # Filter out already completed problems
    remaining_problems = [p for p in problems if p["problem_id"] not in completed_problem_ids]
    logger.info(f"Resuming with {len(remaining_problems)} remaining problems (skipping {len(completed_problem_ids)} completed)")
    
    # Load checkpoint if exists
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    problem_states = load_checkpoint(checkpoint_path, remaining_problems)
    
    if problem_states is None:
        # Initialize fresh problem states
        problem_states = {}
        for p in remaining_problems:
            problem_states[p["problem_id"]] = ProblemState(p)
    
    stage_queues = defaultdict(list)
    completed_problems = []
    
    # Populate stage queues based on current states
    for pid, state in problem_states.items():
        if state.stage != ProblemStage.COMPLETED:
            stage_queues[state.stage].append(pid)
    
    # If no problems in queues (fresh start), initialize
    if not any(stage_queues.values()):
        for pid in problem_states:
            stage_queues[ProblemStage.INITIAL_GENERATION].append(pid)
    
    # Define feedback and revision stages (reduced to 3)
    feedback_stages = [ProblemStage.FEEDBACK_1, ProblemStage.FEEDBACK_2, ProblemStage.FEEDBACK_3]
    revision_stages = [ProblemStage.REVISION_1, ProblemStage.REVISION_2, ProblemStage.REVISION_3]
    check_stages = [ProblemStage.CHECK_1, ProblemStage.CHECK_2, ProblemStage.CHECK_3]
    
    checkpoint_counter = 0
    
    while any(stage_queues[s] for s in ProblemStage if s != ProblemStage.COMPLETED):
        
        # Initial generation with retry for boxed answer
        if stage_queues[ProblemStage.INITIAL_GENERATION]:
            logger.info(f"Generating initial solutions for {len(stage_queues[ProblemStage.INITIAL_GENERATION])} problems")
            
            problems_needing_generation = list(stage_queues[ProblemStage.INITIAL_GENERATION])
            stage_queues[ProblemStage.INITIAL_GENERATION].clear()
            
            retry_count = 0
            while problems_needing_generation and retry_count < config.max_boxed_answer_retries:
                batch_prompts = []
                batch_ids = []
                
                for pid in problems_needing_generation:
                    state = problem_states[pid]
                    # Only generate if dialogue is empty (not resuming mid-generation)
                    if not state.dialogue:
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT_STUDENT},
                            {"role": "user", "content": f"{state.problem}\n\nSolve this step-by-step and put your final answer in \\boxed{{}}."}
                        ]
                        state.dialogue = messages.copy()
                    else:
                        # Retry with emphasis on boxed answer
                        messages = state.dialogue.copy()
                        messages.append({
                            "role": "user", 
                            "content": "Please provide your solution again, making sure to put your final answer in \\boxed{}."
                        })
                    
                    prompt = model_manager.format_student_prompt(messages)
                    batch_prompts.append(prompt)
                    batch_ids.append(pid)
                
                if batch_prompts:
                    # Process in smaller chunks to maximize GPU utilization
                    all_solutions = []
                    chunk_size = config.batch_size * config.num_gpus
                    
                    for i in range(0, len(batch_prompts), chunk_size):
                        chunk = batch_prompts[i:i+chunk_size]
                        solutions = await model_manager.generate_student_batch(chunk, "initial_generation")
                        all_solutions.extend(solutions)
                        logger.info(f"Generated batch {i//chunk_size + 1}/{(len(batch_prompts)-1)//chunk_size + 1}")
                    
                    # Check which problems need retry
                    still_need_generation = []
                    
                    for pid, solution in zip(batch_ids, all_solutions):
                        state = problem_states[pid]
                        
                        if has_boxed_answer(solution):
                            # Success - has boxed answer
                            state.current_solution = solution
                            if retry_count > 0:
                                # Add retry message to dialogue if this was a retry
                                state.dialogue.append({
                                    "role": "user", 
                                    "content": "Please provide your solution again, making sure to put your final answer in \\boxed{}."
                                })
                            state.dialogue.append({"role": "assistant", "content": solution})
                            state.stage = ProblemStage.INITIAL_CHECK
                            stage_queues[ProblemStage.INITIAL_CHECK].append(pid)
                        else:
                            # Need retry
                            state.boxed_answer_retries += 1
                            if state.boxed_answer_retries < config.max_boxed_answer_retries:
                                still_need_generation.append(pid)
                                logger.warning(f"Problem {pid}: No boxed answer found, will retry (attempt {state.boxed_answer_retries + 1})")
                            else:
                                # Max retries reached, mark as failed
                                logger.error(f"Problem {pid}: Failed to generate boxed answer after {config.max_boxed_answer_retries} attempts")
                                state.solved = False
                                state.stage = ProblemStage.COMPLETED
                                completed_problems.append(pid)
                    
                    problems_needing_generation = still_need_generation
                    retry_count += 1
                    
                    if problems_needing_generation:
                        logger.info(f"{len(problems_needing_generation)} problems need retry for boxed answer")
        
        # Check answers using teacher model
        check_requests = []
        for stage in [ProblemStage.INITIAL_CHECK] + check_stages:
            for pid in stage_queues[stage]:
                state = problem_states[pid]
                check_requests.append((pid, state.current_solution, state.expected_answer))
            stage_queues[stage].clear()
        
        if check_requests:
            logger.info(f"Checking {len(check_requests)} answers with teacher model")
            check_results = await model_manager.generate_teacher_batch(check_requests, "check_answer")
            
            for pid, _, _ in check_requests:
                state = problem_states[pid]
                is_correct = check_results.get(pid, False)
                
                if is_correct:
                    # Solution is correct - add final confirmation step
                    if not state.final_confirmation_added:
                        state.stage = ProblemStage.FINAL_CONFIRMATION
                        stage_queues[ProblemStage.FINAL_CONFIRMATION].append(pid)
                    else:
                        # Final confirmation already added, complete
                        state.solved = True
                        state.stage = ProblemStage.COMPLETED
                        completed_problems.append(pid)
                        logger.info(f"Problem {pid}: Solved after {state.revisions} revisions")
                else:
                    # Move to next feedback stage based on current stage and revision count
                    if state.stage == ProblemStage.INITIAL_CHECK:
                        if state.revisions < config.max_revisions:
                            state.stage = ProblemStage.FEEDBACK_1
                            stage_queues[ProblemStage.FEEDBACK_1].append(pid)
                        else:
                            # Max revisions reached
                            state.solved = False
                            state.stage = ProblemStage.COMPLETED
                            completed_problems.append(pid)
                            logger.warning(f"Problem {pid}: Failed after {config.max_revisions} revisions")
                    else:
                        # Find which check stage we're at and move to corresponding feedback
                        for i, check_stage in enumerate(check_stages):
                            if state.stage == check_stage:
                                if state.revisions < config.max_revisions:
                                    if i + 1 < len(feedback_stages):  # Make sure we don't exceed available feedback stages
                                        state.stage = feedback_stages[i + 1]
                                        stage_queues[feedback_stages[i + 1]].append(pid)
                                    else:
                                        # No more feedback stages available
                                        state.solved = False
                                        state.stage = ProblemStage.COMPLETED
                                        completed_problems.append(pid)
                                        logger.warning(f"Problem {pid}: Failed after {config.max_revisions} revisions")
                                else:
                                    # Max revisions reached
                                    state.solved = False
                                    state.stage = ProblemStage.COMPLETED
                                    completed_problems.append(pid)
                                    logger.warning(f"Problem {pid}: Failed after {config.max_revisions} revisions")
                                break
        
        # Handle final confirmation stage
        if stage_queues[ProblemStage.FINAL_CONFIRMATION]:
            logger.info(f"Generating final confirmations for {len(stage_queues[ProblemStage.FINAL_CONFIRMATION])} solved problems")
            
            confirmation_prompts = []
            confirmation_ids = []
            
            for pid in stage_queues[ProblemStage.FINAL_CONFIRMATION]:
                state = problem_states[pid]
                
                # Create prompt for student to generate final confirmation
                messages = state.dialogue.copy()
                messages.append({"role": "user", "content": FINAL_CONFIRMATION_GENERATION_PROMPT})
                
                prompt = model_manager.format_student_prompt(messages)
                confirmation_prompts.append(prompt)
                confirmation_ids.append(pid)
            
            # Generate confirmations using student model
            all_confirmations = []
            chunk_size = config.batch_size * config.num_gpus
            
            for i in range(0, len(confirmation_prompts), chunk_size):
                chunk = confirmation_prompts[i:i+chunk_size]
                confirmations = await model_manager.generate_student_batch(chunk, "final_confirmation")
                all_confirmations.extend(confirmations)
            
            # Process confirmations
            for pid, confirmation in zip(confirmation_ids, all_confirmations):
                state = problem_states[pid]
                
                # Check if confirmation has [NO_REV] flag
                if not has_no_revision_flag(confirmation):
                    # If student didn't add [NO_REV], append it
                    confirmation = confirmation.strip() + " [NO_REV]"
                    logger.debug(f"Problem {pid}: Added missing [NO_REV] flag to final confirmation")
                
                # Add to dialogue with standardized prompt for finetuning
                state.dialogue.append({"role": "user", "content": STANDARDIZED_SELF_REVISION_PROMPT})
                state.dialogue.append({"role": "assistant", "content": confirmation})
                state.final_confirmation_added = True
                state.solved = True
                state.stage = ProblemStage.COMPLETED
                completed_problems.append(pid)
                logger.info(f"Problem {pid}: Generated final confirmation")
            
            stage_queues[ProblemStage.FINAL_CONFIRMATION].clear()
        
        # Generate feedback
        feedback_requests = []
        for stage in feedback_stages:
            for pid in stage_queues[stage]:
                state = problem_states[pid]
                feedback_requests.append((
                    pid, 
                    state.problem, 
                    state.current_solution,
                    state.expected_answer
                ))
            stage_queues[stage].clear()
        
        if feedback_requests:
            logger.info(f"Generating feedback for {len(feedback_requests)} problems")
            feedback_results = await model_manager.generate_teacher_batch(feedback_requests, "feedback")
            
            for pid, _, _, _ in feedback_requests:
                state = problem_states[pid]
                feedback_response = feedback_results.get(pid, "ERROR_TYPE: general\nFEEDBACK: Please review your solution.")
                error_type, feedback = extract_feedback_components(feedback_response)
                
                state.pending_feedback = feedback
                state.pending_error_type = error_type
                
                # Always move to corresponding revision stage
                for i, feedback_stage in enumerate(feedback_stages):
                    if state.stage == feedback_stage:
                        state.stage = revision_stages[i]
                        stage_queues[revision_stages[i]].append(pid)
                        break
        
        # Generate revisions with feedback checking
        for revision_stage_idx, revision_stage in enumerate(revision_stages):
            if not stage_queues[revision_stage]:
                continue
                
            problems_needing_revision = list(stage_queues[revision_stage])
            stage_queues[revision_stage].clear()
            
            # Keep trying until students follow feedback or max attempts reached
            for attempt in range(config.max_revision_attempts):
                if not problems_needing_revision:
                    break
                    
                logger.info(f"Generating revisions for {len(problems_needing_revision)} problems (attempt {attempt + 1})")
                
                revision_prompts = []
                revision_ids = []
                
                for pid in problems_needing_revision:
                    state = problem_states[pid]
                    
                    # Store previous solution for checking
                    state.previous_solution = state.current_solution
                    
                    # Create revision prompt with feedback
                    revision_prompt = DiffusionRevisionSystem.create_revision_prompt(
                        state.pending_feedback, 
                        state.pending_error_type
                    )
                    
                    messages = state.dialogue.copy()
                    messages.append({"role": "user", "content": revision_prompt})
                    
                    prompt = model_manager.format_student_prompt(messages)
                    revision_prompts.append(prompt)
                    revision_ids.append(pid)
                
                # Generate revisions using student model
                all_revisions = []
                chunk_size = config.batch_size * config.num_gpus
                
                for i in range(0, len(revision_prompts), chunk_size):
                    chunk = revision_prompts[i:i+chunk_size]
                    revisions = await model_manager.generate_student_batch(chunk, "revision")
                    all_revisions.extend(revisions)
                
                # Check if revisions addressed feedback
                addressed_checks = []
                for pid, revision in zip(revision_ids, all_revisions):
                    state = problem_states[pid]
                    addressed_checks.append((
                        pid,
                        state.previous_solution,
                        revision,
                        state.pending_feedback
                    ))
                
                addressed_results = await model_manager.generate_teacher_batch(addressed_checks, "check_revision")
                
                # Process results
                still_need_revision = []
                
                for pid, revision in zip(revision_ids, all_revisions):
                    state = problem_states[pid]
                    addressed = addressed_results.get(pid, False)
                    
                    # Store the revision attempt
                    revision_attempt = RevisionAttempt(
                        revision_text=revision,
                        addressed_feedback=addressed,
                        revision_stage=state.revisions,
                        error_type=state.pending_error_type
                    )
                    state.revision_attempts_history.append(revision_attempt)
                    
                    # Accept if addressed
                    if addressed:
                        # Accept the revision
                        state.current_solution = revision
                        state.dialogue.append({"role": "user", "content": f"Teacher feedback: {state.pending_feedback}"})
                        state.dialogue.append({"role": "assistant", "content": revision})
                        state.revisions += 1
                        state.error_types_addressed.append(state.pending_error_type)
                        
                        # Move to check stage
                        state.stage = check_stages[revision_stage_idx]
                        stage_queues[check_stages[revision_stage_idx]].append(pid)
                        
                        logger.debug(f"Problem {pid}: Revision accepted (addressed feedback)")
                    else:
                        # Need to retry
                        state.revision_attempts += 1
                        if state.revision_attempts < config.max_revision_attempts:
                            still_need_revision.append(pid)
                            logger.debug(f"Problem {pid}: Revision rejected (didn't address feedback), will retry (attempt {state.revision_attempts})")
                        else:
                            # Max attempts reached - DO NOT accept poor revisions
                            logger.warning(f"Problem {pid}: Failed to produce acceptable revision after {config.max_revision_attempts} attempts")
                            
                            # Mark as failed due to poor revisions
                            state.failed_due_to_poor_revisions = True
                            state.stage = ProblemStage.COMPLETED
                            completed_problems.append(pid)
                            
                            # Add a note in the dialogue about the failure
                            state.dialogue.append({"role": "user", "content": f"Teacher feedback: {state.pending_feedback}"})
                            state.dialogue.append({"role": "assistant", "content": "[Unable to adequately address the feedback after multiple attempts]"})
                
                problems_needing_revision = still_need_revision
                
                if problems_needing_revision:
                    logger.info(f"{len(problems_needing_revision)} revisions need retry")
        
        # Save completed problems
        for pid in completed_problems:
            state = problem_states[pid]
            result = {
                "problem_id": pid,
                "problem": state.problem,
                "expected_answer": state.expected_answer,
                "dialogue": state.dialogue,
                "num_revisions": state.revisions,
                "final_correct": state.solved,
                "error_types_addressed": state.error_types_addressed,
                "revision_attempts_history": [
                    {
                        "revision_text": attempt.revision_text,
                        "addressed_feedback": attempt.addressed_feedback,
                        "revision_stage": attempt.revision_stage,
                        "error_type": attempt.error_type
                    }
                    for attempt in state.revision_attempts_history
                ],
                "failed_due_to_poor_revisions": state.failed_due_to_poor_revisions,
                "final_confirmation_added": state.final_confirmation_added
            }
            
            result_path = os.path.join(output_dir, f"{pid}.json")
            with open(result_path, "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved problem {pid}: correct={result['final_correct']}, revisions={result['num_revisions']}")
            
            # Remove from problem_states to free memory
            del problem_states[pid]
            checkpoint_counter += 1
        
        completed_problems.clear()
        
        # Save checkpoint periodically
        if checkpoint_counter >= config.checkpoint_interval:
            save_checkpoint(problem_states, checkpoint_path)
            checkpoint_counter = 0
        
        active_count = sum(len(stage_queues[s]) for s in ProblemStage if s != ProblemStage.COMPLETED)
        completed_count = len(completed_problem_ids) + len([f for f in os.listdir(output_dir) if f.endswith('.json') and not f.startswith('generation_')])
        logger.info(f"Progress: {active_count} active, {completed_count}/{len(problems)} completed")
    
    # Clean up checkpoint file when done
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("Removed checkpoint file after completion")
    
    # Load all results for return
    results = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.json') and not filename.startswith('generation_'):
            with open(os.path.join(output_dir, filename), 'r') as f:
                results.append(json.load(f))
    
    return results

def prepare_dataset(config: Config):
    logger.info("Loading SynthLabsAI/Big-Math-RL-Verified dataset...")
    
    dataset = load_dataset(config.dataset_name, split="train")
    df = pd.DataFrame(dataset)
    
    df = df.rename(columns={"answer": "expected_answer"})
    
    logger.info(f"Dataset loaded with {len(df)} problems")
    
    if 'llama8b_solve_rate' in df.columns:
        original_size = len(df)
        df = df[(df['llama8b_solve_rate'] >= config.min_solve_rate) & 
                (df['llama8b_solve_rate'] <= config.max_solve_rate)]
        logger.info(f"Filtered by solve rate [{config.min_solve_rate}-{config.max_solve_rate}]: {len(df)} problems (from {original_size})")
    else:
        logger.warning("No 'llama8b_solve_rate' column found, skipping filter")
    
    logger.info(f"Dataset size before deduplication: {len(df)}")
    df = df.drop_duplicates(subset=['problem'], keep='first')
    logger.info(f"Dataset size after deduplication: {len(df)}")
    
    if len(df) > config.num_samples:
        df = df.sample(n=config.num_samples, random_state=42)
    
    train_size = int(len(df) * config.train_ratio)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    def format_problems(df, id_prefix):
        problems = []
        for idx, row in df.iterrows():
            problems.append({
                "problem_id": f"{id_prefix}_{idx}",
                "problem": row['problem'],
                "expected_answer": str(row['expected_answer'])
            })
        return problems
    
    train_problems = format_problems(train_df, "train")
    test_problems = format_problems(test_df, "test")
    
    os.makedirs("data", exist_ok=True)
    with open("data/train_problems.json", "w", encoding='utf-8') as f:
        json.dump(train_problems, f, indent=2, ensure_ascii=False)
    with open("data/test_problems.json", "w", encoding='utf-8') as f:
        json.dump(test_problems, f, indent=2, ensure_ascii=False)
    
    return train_problems, test_problems

async def main():
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    config = Config()
    
    # Check command line arguments
    import sys
    num_problems = 50000  # Default for testing
    if len(sys.argv) > 1:
        try:
            num_problems = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid argument: {sys.argv[1]}. Please provide a number.")
            return
    
    logger.info(f"Processing {num_problems} problems using {config.num_gpus} GPUs with dynamic model loading")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    if os.path.exists("data/train_problems.json"):
        logger.info("Loading existing dataset...")
        with open("data/train_problems.json", "r", encoding='utf-8') as f:
            train_problems = json.load(f)
    else:
        logger.info("Preparing new dataset...")
        train_problems, test_problems = prepare_dataset(config)
    
    # Use sequential sampling instead of random to avoid missing problems
    problems_to_process = train_problems[:num_problems]
    
    logger.info("Initializing dynamic model manager...")
    model_manager = DynamicModelManager(config)
    
    try:
        logger.info("Starting diffusion-inspired revision trace generation...")
        results = await process_problems_pipeline(
            problems_to_process, 
            model_manager,
            config,
            config.output_dir
        )
        
        # Generate detailed summary statistics
        max_revisions_seen = max((r['num_revisions'] for r in results), default=0)
        solved_at = {i: 0 for i in range(max_revisions_seen + 1)}
        unsolved_at = {i: 0 for i in range(max_revisions_seen + 1)}
        error_type_counts = {"conceptual": 0, "structural": 0, "computational": 0, "interpretation": 0, "general": 0}
        total_revision_attempts = 0
        total_failed_attempts = 0
        problems_failed_due_to_poor_revisions = 0
        problems_with_final_confirmation = 0
        
        for r in results:
            rev = r['num_revisions']
            if r['final_correct']:
                solved_at[rev] += 1
            else:
                unsolved_at[rev] += 1
                if r.get('failed_due_to_poor_revisions', False):
                    problems_failed_due_to_poor_revisions += 1
            
            if r.get('final_confirmation_added', False):
                problems_with_final_confirmation += 1
            
            # Count error types addressed
            for error_type in r.get('error_types_addressed', []):
                if error_type in error_type_counts:
                    error_type_counts[error_type] += 1
            
            # Count revision attempts
            for attempt in r.get('revision_attempts_history', []):
                total_revision_attempts += 1
                if not attempt['addressed_feedback']:
                    total_failed_attempts += 1
        
        stats = {
            "total_problems": len(results),
            "solved_no_revision": solved_at.get(0, 0),
            "solved_with_revisions": sum(solved_at.values()) - solved_at.get(0, 0),
            "unsolved": sum(unsolved_at.values()),
            "avg_revisions": sum(r['num_revisions'] for r in results) / len(results) if results else 0,
            "revision_distribution": {
                f"{i}_revisions": solved_at.get(i, 0) + unsolved_at.get(i, 0) for i in range(max_revisions_seen + 1)
            },
            "solved_at_revisions": solved_at,
            "unsolved_at_revisions": unsolved_at,
            "error_types_addressed": error_type_counts,
            "total_revision_attempts": total_revision_attempts,
            "total_failed_attempts": total_failed_attempts,
            "revision_success_rate": (total_revision_attempts - total_failed_attempts) / total_revision_attempts if total_revision_attempts > 0 else 0,
            "problems_failed_due_to_poor_revisions": problems_failed_due_to_poor_revisions,
            "problems_with_final_confirmation": problems_with_final_confirmation,
            "revision_addressed_distribution": {
                "addressed": sum(1 for r in results for a in r.get('revision_attempts_history', []) if a['addressed_feedback']),
                "not_addressed": sum(1 for r in results for a in r.get('revision_attempts_history', []) if not a['addressed_feedback'])
            },
            "gpu_utilization": f"{config.num_gpus} GPUs with dynamic model loading, batch size {config.batch_size} per GPU"
        }
        
        logger.info(f"Generation complete! Statistics:\n{json.dumps(stats, indent=2)}")
        
        with open(os.path.join(config.output_dir, "generation_summary.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        # Save enhanced finetuning dataset with converted dialogue
        with open("finetuning_dataset.jsonl", "w", encoding='utf-8') as f:
            for result in results:
                # Only include if dialogue contains actual revisions (not failure placeholders)
                if not result.get('failed_due_to_poor_revisions', False) or result['num_revisions'] > 0:
                    # Convert dialogue to use standardized prompts
                    converted_dialogue = convert_dialogue_for_finetuning(result["dialogue"])
                    
                    finetuning_entry = {
                        "dialogue": converted_dialogue,
                        "problem_id": result["problem_id"],
                        "problem": result["problem"],
                        "expected_answer": result["expected_answer"],
                        "num_revisions": result["num_revisions"],
                        "final_correct": result["final_correct"],
                        "error_types_addressed": result.get("error_types_addressed", []),
                        "has_final_confirmation": result.get("final_confirmation_added", False)
                    }
                    f.write(json.dumps(finetuning_entry, ensure_ascii=False) + "\n")
        
        logger.info(f"Enhanced finetuning dataset saved to finetuning_dataset.jsonl")
        
        # Generate DPO dataset that matches finetuning format exactly
        dpo_pairs = []
        
        # Collect successful and failed traces with their converted dialogues
        successful_problems = []
        failed_problems = []
        
        for result in results:
            # Convert dialogue to use standardized prompts (same as finetuning)
            converted_dialogue = convert_dialogue_for_finetuning(result["dialogue"])
            
            # Skip if conversion failed or dialogue is too short
            if not converted_dialogue or len(converted_dialogue) < 3:
                continue
            
            # Extract prompt: system + initial user/problem (first two messages)
            if len(converted_dialogue) >= 2 and converted_dialogue[0]["role"] == "system" and converted_dialogue[1]["role"] == "user":
                prompt_messages = converted_dialogue[:2]  # System + initial problem
                
                # Continuation: everything after the initial prompt
                continuation = converted_dialogue[2:]
                
                trace_info = {
                    "prompt": prompt_messages,
                    "continuation": continuation,
                    "full_dialogue": converted_dialogue,  # For internal use
                    "problem_id": result["problem_id"],
                    "num_revisions": result["num_revisions"],
                    "revision_attempts": result.get("revision_attempts_history", [])
                }
                
                if result["final_correct"] and not result.get("failed_due_to_poor_revisions", False):
                    # Verify continuation ends with [NO_REV]
                    if continuation and continuation[-1]["role"] == "assistant" and not has_no_revision_flag(continuation[-1]["content"]):
                        continuation[-1]["content"] = continuation[-1]["content"].strip() + " [NO_REV]"
                    successful_problems.append(trace_info)
                else:
                    failed_problems.append(trace_info)
        
        # Create DPO pairs
        for success in successful_problems:
            # Chosen: The successful continuation
            chosen_continuation = success["continuation"]
            
            # Generate multiple rejected continuations for this problem
            rejected_continuations = []
            
            # Type 1: Use actual failed problems (same prompt if possible)
            for failed in failed_problems:
                if failed["prompt"] == success["prompt"]:  # Same problem
                    rejected_continuations.append({
                        "continuation": failed["continuation"],
                        "reason": "failed_to_solve"
                    })
                    break  # Use at most one to avoid duplication
            
            # Type 2: Create rejected by including poor revision attempts
            if success["revision_attempts"]:
                for i, attempt in enumerate(success["revision_attempts"]):
                    if not attempt["addressed_feedback"]:
                        # Build rejected continuation up to this poor revision
                        rejected_continuation = []
                        # Copy up to this revision stage
                        msg_count = 1 + (attempt["revision_stage"] * 2)  # Initial solution + (revisions * 2)
                        rejected_continuation = success["continuation"][:msg_count]
                        
                        # Add the poor revision
                        rejected_continuation.append({
                            "role": "user",
                            "content": STANDARDIZED_SELF_REVISION_PROMPT
                        })
                        rejected_continuation.append({
                            "role": "assistant",
                            "content": attempt["revision_text"]
                        })
                        
                        # End with a premature or incorrect [NO_REV]
                        rejected_continuation.append({
                            "role": "user",
                            "content": STANDARDIZED_SELF_REVISION_PROMPT
                        })
                        rejected_continuation.append({
                            "role": "assistant",
                            "content": "I've reviewed but think it's fine, though there might be an issue. [NO_REV]"  # Incorrect confidence
                        })
                        
                        rejected_continuations.append({
                            "continuation": rejected_continuation,
                            "reason": "poor_revision_attempt"
                        })
                        break  # Use first poor attempt
            
            # Type 3: Missing [NO_REV] flag (format error)
            format_error_continuation = [msg.copy() for msg in chosen_continuation]
            if format_error_continuation and format_error_continuation[-1]["role"] == "assistant":
                format_error_continuation[-1]["content"] = re.sub(r'\s*\[NO_REV\]', '', format_error_continuation[-1]["content"])
                rejected_continuations.append({
                    "continuation": format_error_continuation,
                    "reason": "missing_no_rev_flag"
                })
            
            # Type 4: Premature stopping (truncate and add early [NO_REV] with wrong answer)
            if len(chosen_continuation) > 4:  #
                early_stop_continuation = chosen_continuation[:3]  # e.g., initial solution + one revision prompt + response
                early_stop_continuation.append({
                    "role": "user",
                    "content": STANDARDIZED_SELF_REVISION_PROMPT
                })
                early_stop_continuation.append({
                    "role": "assistant",
                    "content": "My solution looks correct to me. I don't see any errors. [NO_REV]"  # Premature with potentially wrong answer
                })
                rejected_continuations.append({
                    "continuation": early_stop_continuation,
                    "reason": "premature_stopping"
                })
            
            # Create DPO pairs (only if there are rejected)
            for rejected in rejected_continuations:
                dpo_entry = {
                    "prompt": success["prompt"],  # System + initial problem
                    "chosen": chosen_continuation,  # Successful continuation (no repetition)
                    "rejected": rejected["continuation"],  # Failed continuation (no repetition)
                    "metadata": {
                        "problem_id": success["problem_id"],
                        "chosen_revisions": success["num_revisions"],
                        "rejection_reason": rejected["reason"]
                    }
                }
                dpo_pairs.append(dpo_entry)
        
        # Save DPO dataset
        if dpo_pairs:
            with open("dpo_dataset.jsonl", "w", encoding='utf-8') as f:
                for pair in dpo_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
            logger.info(f"DPO dataset saved with {len(dpo_pairs)} chosen/rejected pairs")
            
            # Log statistics
            rejection_reasons = {}
            for pair in dpo_pairs:
                reason = pair["metadata"]["rejection_reason"]
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            logger.info(f"Rejection reasons: {rejection_reasons}")
        
        # Save filtered datasets
        solved_problems = [r for r in results if r['final_correct']]
        with open("finetuning_dataset_solved.jsonl", "w", encoding='utf-8') as f:
            for result in solved_problems:
                converted_dialogue = convert_dialogue_for_finetuning(result["dialogue"])
                finetuning_entry = {
                    "dialogue": converted_dialogue,
                    "problem_id": result["problem_id"],
                    "problem": result["problem"],
                    "expected_answer": result["expected_answer"],
                    "num_revisions": result["num_revisions"],
                    "final_correct": result["final_correct"],
                    "error_types_addressed": result.get("error_types_addressed", []),
                    "has_final_confirmation": result.get("final_confirmation_added", False)
                }
                f.write(json.dumps(finetuning_entry, ensure_ascii=False) + "\n")
        
        improved_problems = [r for r in results if r['final_correct'] and r['num_revisions'] > 0]
        with open("finetuning_dataset_improved.jsonl", "w", encoding='utf-8') as f:
            for result in improved_problems:
                converted_dialogue = convert_dialogue_for_finetuning(result["dialogue"])
                finetuning_entry = {
                    "dialogue": converted_dialogue,
                    "problem_id": result["problem_id"],
                    "problem": result["problem"],
                    "expected_answer": result["expected_answer"],
                    "num_revisions": result["num_revisions"],
                    "final_correct": result["final_correct"],
                    "error_types_addressed": result.get("error_types_addressed", []),
                    "has_final_confirmation": result.get("final_confirmation_added", False)
                }
                f.write(json.dumps(finetuning_entry, ensure_ascii=False) + "\n")
        
        logger.info(f"Additional filtered datasets created:")
        logger.info(f"  - finetuning_dataset_solved.jsonl: {len(solved_problems)} problems")
        logger.info(f"  - finetuning_dataset_improved.jsonl: {len(improved_problems)} problems")
        logger.info(f"  - dpo_dataset.jsonl: {len(dpo_pairs)} chosen/rejected pairs")
        
    finally:
        # Ensure workers are shut down
        logger.info("Shutting down model manager...")
        model_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())