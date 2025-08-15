import json
import re

def count_words(text):
    """Count words in a text string."""
    # Remove LaTeX commands and special characters, then count words
    # First remove LaTeX commands like \boxed{}, \times, etc.
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    # Remove other special characters but keep alphanumeric
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def analyze_dpo_jsonl(filename):
    """Analyze DPO JSONL file to calculate average word counts."""
    prompt_word_counts = []
    chosen_response_word_counts = []
    rejected_response_word_counts = []
    
    # Track different types of responses
    chosen_initial_response_counts = []
    chosen_revision_response_counts = []
    rejected_initial_response_counts = []
    rejected_revision_response_counts = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            if line.strip():  # Skip empty lines
                try:
                    entry = json.loads(line)
                    
                    # Count prompt words
                    prompt_messages = entry.get('prompt', [])
                    prompt_text = ""
                    for msg in prompt_messages:
                        prompt_text += msg.get('content', '') + " "
                    prompt_word_count = count_words(prompt_text)
                    prompt_word_counts.append(prompt_word_count)
                    
                    # Count chosen response words
                    chosen_messages = entry.get('chosen', [])
                    for i, msg in enumerate(chosen_messages):
                        if msg.get('role') == 'assistant':
                            content = msg.get('content', '')
                            word_count = count_words(content)
                            chosen_response_word_counts.append(word_count)
                            
                            # Check if it's initial response or revision
                            if i == 0:  # First assistant response
                                chosen_initial_response_counts.append(word_count)
                            else:  # Revision response
                                chosen_revision_response_counts.append(word_count)
                    
                    # Count rejected response words
                    rejected_messages = entry.get('rejected', [])
                    for i, msg in enumerate(rejected_messages):
                        if msg.get('role') == 'assistant':
                            content = msg.get('content', '')
                            word_count = count_words(content)
                            rejected_response_word_counts.append(word_count)
                            
                            # Check if it's initial response or revision
                            if i == 0:  # First assistant response
                                rejected_initial_response_counts.append(word_count)
                            else:  # Revision response
                                rejected_revision_response_counts.append(word_count)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                except Exception as e:
                    print(f"Error processing entry on line {line_num}: {e}")
    
    # Calculate and display statistics
    print("=== DPO JSONL ANALYSIS RESULTS ===\n")
    
    print("PROMPT STATISTICS:")
    print(f"Total prompts analyzed: {len(prompt_word_counts)}")
    print(f"Average words per prompt: {sum(prompt_word_counts) / len(prompt_word_counts):.2f}" if prompt_word_counts else "N/A")
    print(f"Min prompt words: {min(prompt_word_counts)}" if prompt_word_counts else "N/A")
    print(f"Max prompt words: {max(prompt_word_counts)}" if prompt_word_counts else "N/A")
    
    print("\nCHOSEN RESPONSE STATISTICS:")
    print(f"Total chosen responses: {len(chosen_response_word_counts)}")
    print(f"Average words per chosen response (all): {sum(chosen_response_word_counts) / len(chosen_response_word_counts):.2f}" if chosen_response_word_counts else "N/A")
    print(f"  - Initial responses: {len(chosen_initial_response_counts)}")
    print(f"  - Average words (initial): {sum(chosen_initial_response_counts) / len(chosen_initial_response_counts):.2f}" if chosen_initial_response_counts else "N/A")
    print(f"  - Revision responses: {len(chosen_revision_response_counts)}")
    print(f"  - Average words (revision): {sum(chosen_revision_response_counts) / len(chosen_revision_response_counts):.2f}" if chosen_revision_response_counts else "N/A")
    
    print("\nREJECTED RESPONSE STATISTICS:")
    print(f"Total rejected responses: {len(rejected_response_word_counts)}")
    print(f"Average words per rejected response (all): {sum(rejected_response_word_counts) / len(rejected_response_word_counts):.2f}" if rejected_response_word_counts else "N/A")
    print(f"  - Initial responses: {len(rejected_initial_response_counts)}")
    print(f"  - Average words (initial): {sum(rejected_initial_response_counts) / len(rejected_initial_response_counts):.2f}" if rejected_initial_response_counts else "N/A")
    print(f"  - Revision responses: {len(rejected_revision_response_counts)}")
    print(f"  - Average words (revision): {sum(rejected_revision_response_counts) / len(rejected_revision_response_counts):.2f}" if rejected_revision_response_counts else "N/A")
    
    # Additional comparative statistics
    print("\nCOMPARATIVE ANALYSIS:")
    if chosen_response_word_counts and rejected_response_word_counts:
        avg_chosen = sum(chosen_response_word_counts) / len(chosen_response_word_counts)
        avg_rejected = sum(rejected_response_word_counts) / len(rejected_response_word_counts)
        diff = avg_chosen - avg_rejected
        percent_diff = (diff / avg_rejected) * 100 if avg_rejected > 0 else 0
        print(f"Chosen responses are on average {diff:.2f} words {'longer' if diff > 0 else 'shorter'} ({percent_diff:+.1f}%)")
    
    return {
        'avg_prompt_words': sum(prompt_word_counts) / len(prompt_word_counts) if prompt_word_counts else 0,
        'avg_chosen_words': sum(chosen_response_word_counts) / len(chosen_response_word_counts) if chosen_response_word_counts else 0,
        'avg_rejected_words': sum(rejected_response_word_counts) / len(rejected_response_word_counts) if rejected_response_word_counts else 0
    }

# Enhanced version with more detailed analysis
def analyze_dpo_jsonl_detailed(filename):
    """Analyze DPO JSONL file with additional metrics."""
    entries_data = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            if line.strip():
                try:
                    entry = json.loads(line)
                    entry_stats = {
                        'problem_id': entry.get('metadata', {}).get('problem_id', 'unknown'),
                        'chosen_revisions': entry.get('metadata', {}).get('chosen_revisions', 0),
                        'rejection_reason': entry.get('metadata', {}).get('rejection_reason', 'unknown'),
                        'prompt_words': 0,
                        'chosen_words': [],
                        'rejected_words': [],
                        'has_no_rev': False
                    }
                    
                    # Analyze prompt
                    prompt_text = ""
                    for msg in entry.get('prompt', []):
                        prompt_text += msg.get('content', '') + " "
                    entry_stats['prompt_words'] = count_words(prompt_text)
                    
                    # Analyze chosen responses
                    for msg in entry.get('chosen', []):
                        if msg.get('role') == 'assistant':
                            content = msg.get('content', '')
                            entry_stats['chosen_words'].append(count_words(content))
                            if '[NO_REV]' in content:
                                entry_stats['has_no_rev'] = True
                    
                    # Analyze rejected responses
                    for msg in entry.get('rejected', []):
                        if msg.get('role') == 'assistant':
                            content = msg.get('content', '')
                            entry_stats['rejected_words'].append(count_words(content))
                    
                    entries_data.append(entry_stats)
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
    
    # Analyze by rejection reason
    print("\n=== ANALYSIS BY REJECTION REASON ===")
    rejection_reasons = {}
    for entry in entries_data:
        reason = entry['rejection_reason']
        if reason not in rejection_reasons:
            rejection_reasons[reason] = {
                'count': 0,
                'avg_chosen_words': [],
                'avg_rejected_words': []
            }
        rejection_reasons[reason]['count'] += 1
        if entry['chosen_words']:
            rejection_reasons[reason]['avg_chosen_words'].extend(entry['chosen_words'])
        if entry['rejected_words']:
            rejection_reasons[reason]['avg_rejected_words'].extend(entry['rejected_words'])
    
    for reason, data in rejection_reasons.items():
        print(f"\nRejection Reason: {reason}")
        print(f"  Count: {data['count']}")
        if data['avg_chosen_words']:
            print(f"  Avg chosen words: {sum(data['avg_chosen_words']) / len(data['avg_chosen_words']):.2f}")
        if data['avg_rejected_words']:
            print(f"  Avg rejected words: {sum(data['avg_rejected_words']) / len(data['avg_rejected_words']):.2f}")

# Usage
if __name__ == "__main__":
    filename = "dpo_dataset.jsonl"  # Replace with your JSONL file path
    
    # Basic analysis
    print("Running basic analysis...")
    results = analyze_dpo_jsonl(filename)
    
    # Detailed analysis
    print("\n" + "="*50 + "\n")
    print("Running detailed analysis...")
    analyze_dpo_jsonl_detailed(filename)