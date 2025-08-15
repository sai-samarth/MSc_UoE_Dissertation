import json
import re

def count_words(text):
    """Count words in a text string."""
    # Remove special characters and split by whitespace
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def analyze_jsonl_file(filename):
    """Analyze JSONL file to calculate average word counts."""
    revision_word_counts = []
    final_reflection_word_counts = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                entry = json.loads(line)
                dialogue = entry.get('dialogue', [])
                
                # Track revision count for this entry
                revision_count = 0
                
                for turn in dialogue:
                    if turn['role'] == 'assistant':
                        content = turn['content']
                        
                        # Check if this is a final NO_REV reflection
                        if '[NO_REV]' in content:
                            # Extract the reflection part (before [NO_REV])
                            reflection_part = content.split('[NO_REV]')[0].strip()
                            word_count = count_words(reflection_part)
                            final_reflection_word_counts.append(word_count)
                        else:
                            # This is a revision response
                            # Skip the first assistant response (initial solution)
                            if revision_count > 0:
                                word_count = count_words(content)
                                revision_word_counts.append(word_count)
                            revision_count += 1
    
    # Calculate averages
    avg_revision_words = sum(revision_word_counts) / len(revision_word_counts) if revision_word_counts else 0
    avg_final_reflection_words = sum(final_reflection_word_counts) / len(final_reflection_word_counts) if final_reflection_word_counts else 0
    
    # Print results
    print(f"Total revision responses analyzed: {len(revision_word_counts)}")
    print(f"Average words per revision response: {avg_revision_words:.2f}")
    print(f"\nTotal final reflections ([NO_REV]) analyzed: {len(final_reflection_word_counts)}")
    print(f"Average words per final reflection: {avg_final_reflection_words:.2f}")
    
    return avg_revision_words, avg_final_reflection_words

# Usage
if __name__ == "__main__":
    filename = "/home/saisamarth/dev/dis/pipeline/finetuning_dataset_solved.jsonl"  # Replace with your JSONL file path
    analyze_jsonl_file(filename)