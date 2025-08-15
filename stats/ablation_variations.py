import json
import os
from dataclasses import dataclass, replace
from typing import Optional, List, Dict
import logging

# Import the original evaluation script
from math_eval import (
    Config, Problem, StudentModel, CheckerModel,
    load_problems, save_results, extract_boxed_answer,
    get_last_valid_answer, format_prompt, 
    evaluate_solutions_standard, evaluate_solutions_comprehensive
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Alternative prompts and configurations
ALTERNATIVE_REVISION_PROMPTS = {
    "variant1": """Review your solution carefully. Check for:
- Correct interpretation of the problem
- Mathematical accuracy and logical flow
- Computational errors
- Proper answer formatting

If you find errors, state: (1) the mistake, (2) the correction needed. Then provide the corrected solution.
If your solution is correct, explain why in 2 sentences and confirm no revision is needed by adding [NO_REV] at the end.""",
    
    "variant2": """Double-check your work above. Examine:
• Understanding of the question
• Step-by-step reasoning validity
• Arithmetic calculations
• Answer completeness

Found a mistake? Briefly describe it and how to fix it (2 sentences), then give your revised answer.
Solution is correct? Justify why (2 sentences) and confirm no revision is needed by adding [NO_REV] at the end.""",
    
    "variant3": """Critically analyze your solution for potential issues:
1. Did you understand the problem correctly?
2. Is your mathematical reasoning sound?
3. Are all calculations accurate?
4. Is the final answer properly formatted?

If there's an error, explain what went wrong and how you'll fix it, then provide the correction.
If everything is correct, briefly confirm this and confirm no revision is needed by adding [NO_REV] at the end."""
}

NON_COT_PROMPTS = {
    "direct": """Solve this problem and put your final answer in \\boxed{}.""",
    
    "minimal": """What is the answer? Put it in \\boxed{}.""",
    
    "no_steps": """Calculate the answer directly without showing work. Final answer in \\boxed{}."""
}

TERMINATION_METHODS = {
    "keyword": {
        "termination_text": "STOP_REVISION",
        "check_function": lambda text: "STOP_REVISION" in text if text else False
    },
    "punctuation": {
        "termination_text": "!!DONE!!",
        "check_function": lambda text: "!!DONE!!" in text if text else False
    },
    "structured": {
        "termination_text": "<no_revision_needed/>",
        "check_function": lambda text: "<no_revision_needed/>" in text if text else False
    }
}

# Default revision prompt (same as original)
DEFAULT_REVISION_PROMPT = """Please analyze your previous solution thoroughly and identify any potential errors or areas for improvement. Carefully check your:
- Problem interpretation and understanding
- Mathematical reasoning and logic
- Calculations and arithmetic
- Final answer format and completeness

If you find any mistakes, first explain in 2 sentences: (1) what specific error you made, (2) how you'll correct it. Then provide your revised solution.
If you're confident your solution is correct after careful review, briefly explain why (2 sentences) and confirm no revision is needed by adding [NO_REV] at the end."""

@dataclass
class RobustnessTestConfig:
    """Configuration for robustness testing"""
    base_config: Config
    test_name: str
    
    # Test variations
    use_alternative_revision_prompt: Optional[str] = None  # key from ALTERNATIVE_REVISION_PROMPTS
    use_non_cot: Optional[str] = None  # key from NON_COT_PROMPTS
    use_alternative_termination: Optional[str] = None  # key from TERMINATION_METHODS
    
    # Output
    output_suffix: str = ""

def generate_solutions_with_modifications(
    problems: List[Problem], 
    config: Config,
    revision_prompt: str,
    system_prompt: str,
    initial_instruction: str,
    termination_check_fn
) -> List[Problem]:
    """Modified version of generate_student_solutions with custom prompts"""
    logger.info("=== Generating Student Solutions (Modified) ===")
    
    student = StudentModel(config)
    
    # Process in batches
    for batch_start in range(0, len(problems), config.batch_size):
        batch_end = min(batch_start + config.batch_size, len(problems))
        batch_problems = problems[batch_start:batch_end]
        
        logger.info(f"Processing batch: problems {batch_start+1}-{batch_end}")
        
        # Generate initial solutions
        prompts = []
        for problem in batch_problems:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{problem.question}\n\n{initial_instruction}"}
            ]
            prompts.append(format_prompt(messages))
        
        responses = student.generate_batch(prompts)
        
        for problem, response in zip(batch_problems, responses):
            problem.initial_solution = response
            problem.final_solution = response
            problem.revision_chain.append(response)
        
        # Revision loop
        for revision_round in range(config.max_revisions):
            # Filter active problems
            active_problems = [p for p in batch_problems if not p.stopped_with_no_rev]
            if not active_problems:
                break
            
            logger.info(f"  Revision {revision_round + 1}: {len(active_problems)} problems active")
            
            # Generate revisions
            prompts = []
            for problem in active_problems:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{problem.question}\n\n{initial_instruction}"},
                    {"role": "assistant", "content": problem.final_solution},
                    {"role": "user", "content": revision_prompt}
                ]
                prompts.append(format_prompt(messages))
            
            responses = student.generate_batch(prompts)
            
            for problem, response in zip(active_problems, responses):
                problem.revision_chain.append(response)
                problem.final_solution = response
                
                # Check for termination
                if termination_check_fn(response):
                    problem.stopped_with_no_rev = True
                else:
                    problem.num_revisions += 1
    
    student.unload()
    logger.info("Student generation complete")
    
    return problems

def run_robustness_test(test_config: RobustnessTestConfig):
    """Run a single robustness test"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Robustness Test: {test_config.test_name}")
    logger.info(f"{'='*60}")
    
    # Set up prompts and functions based on test configuration
    revision_prompt = DEFAULT_REVISION_PROMPT
    system_prompt = "You are a student working on mathematical problems. Show your step-by-step reasoning and provide the final answer within \\boxed{}."
    initial_instruction = "Solve this step-by-step and put your final answer in \\boxed{}."
    termination_check_fn = lambda text: "[NO_REV]" in text if text else False
    
    # Apply modifications
    if test_config.use_alternative_revision_prompt:
        # Use alternative revision prompt but keep [NO_REV] termination
        revision_prompt = ALTERNATIVE_REVISION_PROMPTS[test_config.use_alternative_revision_prompt]
        logger.info(f"Using alternative revision prompt: {test_config.use_alternative_revision_prompt}")
    
    if test_config.use_alternative_termination:
        # Keep the same revision prompt prefix but change termination
        term_config = TERMINATION_METHODS[test_config.use_alternative_termination]
        revision_prompt = revision_prompt.replace(
            "[NO_REV]",
            term_config["termination_text"]
        )
        termination_check_fn = term_config["check_function"]
        logger.info(f"Using alternative termination: {test_config.use_alternative_termination}")
    
    if test_config.use_non_cot:
        # Change system prompt and initial instruction for non-CoT
        system_prompt = "You are a student working on mathematical problems. Provide the final answer within \\boxed{}."
        initial_instruction = NON_COT_PROMPTS[test_config.use_non_cot]
        logger.info(f"Using non-CoT mode: {test_config.use_non_cot}")
    
    # Load problems
    problems = load_problems(test_config.base_config)
    
    # Generate solutions with modifications
    problems = generate_solutions_with_modifications(
        problems, 
        test_config.base_config,
        revision_prompt,
        system_prompt,
        initial_instruction,
        termination_check_fn
    )
    
    # Evaluate solutions
    if test_config.base_config.comprehensive_eval:
        metrics = evaluate_solutions_comprehensive(problems, test_config.base_config)
    else:
        metrics = evaluate_solutions_standard(problems, test_config.base_config)
    
    # Add test configuration to metrics
    metrics["test_configuration"] = {
        "test_name": test_config.test_name,
        "alternative_revision_prompt": test_config.use_alternative_revision_prompt,
        "non_cot_mode": test_config.use_non_cot,
        "alternative_termination": test_config.use_alternative_termination
    }
    
    # Save results with unique suffix
    output_dir = f"{test_config.base_config.output_dir}_{test_config.output_suffix}"
    test_config.base_config.output_dir = output_dir
    save_results(problems, metrics, test_config.base_config)
    
    return metrics

def run_all_robustness_tests():
    """Run all robustness tests and compare results"""
    # Base configuration
    base_config = Config(
        num_problems=50,  # Use smaller set for testing
        output_dir="robustness_results"
    )
    
    # Parse command line arguments for base configuration
    import sys
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--lora" and i + 1 < len(args):
            base_config.lora_adapter_path = args[i + 1]
            i += 2
        elif args[i] == "--custom-test" and i + 1 < len(args):  # ADD THIS
            base_config.custom_test_file = args[i + 1]
            i += 2
        elif args[i] == "--num-problems" and i + 1 < len(args):
            base_config.num_problems = int(args[i + 1])
            i += 2
        elif args[i] == "--deepinfra-api-key" and i + 1 < len(args):
            base_config.deepinfra_api_key = args[i + 1]
            i += 2
        elif args[i] == "-eval":
            base_config.comprehensive_eval = True
            i += 1
        else:
            i += 1
    
    all_results = {}
    
    # Test 1: Baseline
    test_config = RobustnessTestConfig(
        base_config=replace(base_config),
        test_name="Baseline",
        output_suffix="baseline"
    )
    all_results["baseline"] = run_robustness_test(test_config)
    
    # Test 2: Alternative revision prompts (keeping [NO_REV] termination)
    for prompt_key in ALTERNATIVE_REVISION_PROMPTS:
        test_config = RobustnessTestConfig(
            base_config=replace(base_config),
            test_name=f"Alternative Revision Prompt - {prompt_key}",
            use_alternative_revision_prompt=prompt_key,
            output_suffix=f"revision_{prompt_key}"
        )
        all_results[f"revision_{prompt_key}"] = run_robustness_test(test_config)
    
    # Test 3: Non-CoT approaches
    for prompt_key in NON_COT_PROMPTS:
        test_config = RobustnessTestConfig(
            base_config=replace(base_config),
            test_name=f"Non-CoT - {prompt_key}",
            use_non_cot=prompt_key,
            output_suffix=f"non_cot_{prompt_key}"
        )
        all_results[f"non_cot_{prompt_key}"] = run_robustness_test(test_config)
    
    # Test 4: Alternative termination methods (keeping same revision prompt prefix)
    for term_key in TERMINATION_METHODS:
        test_config = RobustnessTestConfig(
            base_config=replace(base_config),
            test_name=f"Alternative Termination - {term_key}",
            use_alternative_termination=term_key,
            output_suffix=f"termination_{term_key}"
        )
        all_results[f"termination_{term_key}"] = run_robustness_test(test_config)
    
    # Generate comparison report
    generate_comparison_report(all_results, base_config.output_dir)

def generate_comparison_report(results: Dict, output_dir: str):
    """Generate a comparison report of all robustness tests"""
    report = {
        "summary": {},
        "detailed_results": results
    }
    
    # Create summary table
    logger.info("\n" + "="*80)
    logger.info("ROBUSTNESS TEST COMPARISON SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Test Name':<30} {'Initial Acc':<12} {'Final Acc':<12} {'Improvement':<12} {'Avg Rev':<10}")
    logger.info("-"*80)
    
    for test_name, metrics in results.items():
        report["summary"][test_name] = {
            "initial_accuracy": metrics["initial_accuracy"],
            "final_accuracy": metrics["final_accuracy"],
            "improvement_rate": metrics["improvement_rate"],
            "avg_revisions": metrics["revision_stats"]["avg_revisions"]
        }
        
        logger.info(
            f"{test_name:<30} "
            f"{metrics['initial_accuracy']:>10.2f}% "
            f"{metrics['final_accuracy']:>10.2f}% "
            f"{metrics['improvement_rate']:>+10.2f}% "
            f"{metrics['revision_stats']['avg_revisions']:>8.2f}"
        )
    
    # Save comparison report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "robustness_comparison.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nComparison report saved to: {report_path}")

def run_single_test():
    """Run a single robustness test based on command line arguments"""
    import sys
    
    # Base configuration
    base_config = Config(
        output_dir="robustness_results"
    )
    
    # Parse arguments
    test_type = None
    test_variant = None
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--lora" and i + 1 < len(args):
            base_config.lora_adapter_path = args[i + 1]
            i += 2
        elif args[i] == "--custom-test" and i + 1 < len(args):  # ADD THIS
            base_config.custom_test_file = args[i + 1]
            i += 2
        elif args[i] == "--num-problems" and i + 1 < len(args):
            base_config.num_problems = int(args[i + 1])
            i += 2
        elif args[i] == "--deepinfra-api-key" and i + 1 < len(args):
            base_config.deepinfra_api_key = args[i + 1]
            i += 2
        elif args[i] == "-eval":
            base_config.comprehensive_eval = True
            i += 1
        elif args[i] == "--test-revision" and i + 1 < len(args):
            test_type = "revision"
            test_variant = args[i + 1]
            i += 2
        elif args[i] == "--test-non-cot" and i + 1 < len(args):
            test_type = "non-cot"
            test_variant = args[i + 1]
            i += 2
        elif args[i] == "--test-termination" and i + 1 < len(args):
            test_type = "termination"
            test_variant = args[i + 1]
            i += 2
        else:
            i += 1
    
    if not test_type:
        logger.error("No test type specified. Use --test-revision, --test-non-cot, or --test-termination")
        return
    
    # Create test configuration
    test_config = RobustnessTestConfig(
        base_config=base_config,
        test_name=f"{test_type}_{test_variant}",
        output_suffix=f"{test_type}_{test_variant}"
    )
    
    if test_type == "revision":
        if test_variant not in ALTERNATIVE_REVISION_PROMPTS:
            logger.error(f"Invalid revision variant: {test_variant}. Choose from: {list(ALTERNATIVE_REVISION_PROMPTS.keys())}")
            return
        test_config.use_alternative_revision_prompt = test_variant
        test_config.test_name = f"Alternative Revision - {test_variant}"
    elif test_type == "non-cot":
        if test_variant not in NON_COT_PROMPTS:
            logger.error(f"Invalid non-CoT variant: {test_variant}. Choose from: {list(NON_COT_PROMPTS.keys())}")
            return
        test_config.use_non_cot = test_variant
        test_config.test_name = f"Non-CoT - {test_variant}"
    elif test_type == "termination":
        if test_variant not in TERMINATION_METHODS:
            logger.error(f"Invalid termination variant: {test_variant}. Choose from: {list(TERMINATION_METHODS.keys())}")
            return
        test_config.use_alternative_termination = test_variant
        test_config.test_name = f"Alternative Termination - {test_variant}"
    
    # Run the test
    metrics = run_robustness_test(test_config)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Test: {test_config.test_name}")
    logger.info(f"Initial Accuracy: {metrics['initial_accuracy']:.2f}%")
    logger.info(f"Final Accuracy: {metrics['final_accuracy']:.2f}%")
    logger.info(f"Improvement: {metrics['improvement_rate']:+.2f}%")
    logger.info(f"Average Revisions: {metrics['revision_stats']['avg_revisions']:.2f}")

if __name__ == "__main__":
    import sys
    
    if "--baseline" in sys.argv:
        # Run only baseline with defaults
        base_config = Config()
        
        # Parse only essential arguments
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            if args[i] == "--lora" and i + 1 < len(args):
                base_config.lora_adapter_path = args[i + 1]
                i += 2
            elif args[i] == "--num-problems" and i + 1 < len(args):
                base_config.num_problems = int(args[i + 1])
                i += 2
            elif args[i] == "--deepinfra-api-key" and i + 1 < len(args):
                base_config.deepinfra_api_key = args[i + 1]
                i += 2
            elif args[i] == "-eval":
                base_config.comprehensive_eval = True
                i += 1
            else:
                i += 1
        
        test_config = RobustnessTestConfig(
            base_config=base_config,
            test_name="Baseline (All Defaults)",
            output_suffix="baseline"
        )
        
        metrics = run_robustness_test(test_config)
        
        logger.info("\n" + "="*60)
        logger.info("BASELINE TEST RESULTS (ALL DEFAULTS)")
        logger.info("="*60)
        logger.info(f"Initial Accuracy: {metrics['initial_accuracy']:.2f}%")
        logger.info(f"Final Accuracy: {metrics['final_accuracy']:.2f}%")
        logger.info(f"Improvement: {metrics['improvement_rate']:+.2f}%")
        logger.info(f"Average Revisions: {metrics['revision_stats']['avg_revisions']:.2f}")        
   
    if "--test-all" in sys.argv:
        run_all_robustness_tests()
    elif any(arg in sys.argv for arg in ["--test-revision", "--test-non-cot", "--test-termination"]):
        run_single_test()
    else:
        print("""
Robustness Testing for Math Evaluation

Usage:
  python robustness_test.py [OPTIONS] [TEST]

Options:
  --lora PATH                   Path to LoRA adapter
  --num-problems N             Number of problems to test
  --deepinfra-api-key KEY      DeepInfra API key
  -eval                        Enable comprehensive evaluation

Test Options (choose one):
  --test-all                   Run all robustness tests
  --test-revision VARIANT      Test alternative revision prompt
                              Variants: variant1, variant2, variant3
  --test-non-cot MODE         Test non-CoT mode
                              Modes: direct, minimal, no_steps
  --test-termination METHOD   Test alternative termination
                              Methods: keyword, punctuation, structured

Examples:
  # Test a specific revision prompt variant
  python robustness_test.py --lora /path/to/lora --test-revision variant1 --num-problems 50
  
  # Test non-CoT mode
  python robustness_test.py --lora /path/to/lora --test-non-cot direct --num-problems 50
  
  # Test alternative termination method
  python robustness_test.py --lora /path/to/lora --test-termination keyword --num-problems 50
  
  # Run all tests
  python robustness_test.py --lora /path/to/lora --test-all --num-problems 50
""")