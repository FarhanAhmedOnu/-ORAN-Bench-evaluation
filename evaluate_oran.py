import json
import ollama
import re
import os

# --- Configuration ---
MODEL_NAME = 'qwen3.5:4b'             # Ensure you have run 'ollama pull mistral'
DATASET_PATH = 'Benchmark/fin_E.json' # Path to your JSONL file
MAX_QUESTIONS = 20                 # Set to None to run the entire file

def evaluate_model():
    print(f"--- ORAN-Bench Evaluation Start ---")
    print(f"Model: {MODEL_NAME}")
    print(f"File:  {DATASET_PATH}")
    
    dataset = []
    
    # Check if file exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: File '{DATASET_PATH}' not found.")
        return

    # 1. Load the dataset (Handling JSON Lines format)
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        dataset.append(json.loads(line))
                    except json.JSONDecodeError:
                        # If a line fails, it might be an empty bracket or malformed
                        continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    total_available = len(dataset)
    if MAX_QUESTIONS:
        dataset = dataset[:MAX_QUESTIONS]
    
    total_to_run = len(dataset)
    print(f"Loaded {total_to_run} questions (out of {total_available} total).\n")

    correct_count = 0
    processed_count = 0

    # 2. Loop through questions
    for i, item in enumerate(dataset):
        # Format per README: [ "Question", ["Opt1", "Opt2", "Opt3", "Opt4"], "CorrectIndex" ]
        question_text = item[0]
        options = item[1]
        correct_answer_index = str(item[2]).strip()

        # Construct the Prompt
        prompt = (
            f"Context: Open Radio Access Network (O-RAN) Technical Specification.\n"
            f"Question: {question_text}\n"
            f"Options:\n"
        )
        for idx, opt in enumerate(options):
            prompt += f"{idx + 1}. {opt}\n"
        
        prompt += "\nTask: Provide ONLY the number (1, 2, 3, or 4) of the correct option. Do not explain."

        # 3. Call Ollama
        try:
            response = ollama.generate(model=MODEL_NAME, prompt=prompt)
            raw_output = response['response'].strip()

            # 4. Extract the first digit found in the model's response
            match = re.search(r'\d', raw_output)
            predicted_index = match.group() if match else None

            # 5. Compare and Log
            is_correct = (predicted_index == correct_answer_index)
            if is_correct:
                correct_count += 1
            
            processed_count += 1
            
            # Print individual result for tracking
            status = "✓" if is_correct else f"✗ (Model said {predicted_index}, Correct was {correct_answer_index})"
            print(f"[{processed_count}/{total_to_run}] {status}")

        except Exception as e:
            print(f"Error on question {i+1}: {e}")

    # 6. Final Results
    if processed_count > 0:
        accuracy = (correct_count / processed_count) * 100
        print("\n" + "="*30)
        print("FINAL RESULTS")
        print("="*30)
        print(f"Total Evaluated: {processed_count}")
        print(f"Correct:          {correct_count}")
        print(f"Accuracy:         {accuracy:.2f}%")
        print("="*30)
    else:
        print("No questions were processed.")

if __name__ == "__main__":
    evaluate_model()