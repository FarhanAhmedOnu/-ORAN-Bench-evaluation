import json
from openai import OpenAI  # Changed from ollama
import re
import os
import time
import sys

# --- Configuration ---
# LM Studio usually loads a specific model ID. 
# If 'Strict Mode' is off in LM Studio server settings, this can be anything.
# If 'Strict Mode' is on, this must match the loaded model ID exactly.
MODEL_NAME = 'qwen/qwen3.5-9b'            

# LM Studio Local Server URL
API_BASE_URL = "http://localhost:1234/v1" 
API_KEY = "lm-studio"                   # Can be anything for local server

DATASET_PATH = 'Benchmark/fin_E.json'   # Path to your JSONL file
MAX_QUESTIONS = 20                      # Set to None to run the entire file
RESULTS_BASE_DIR = "results"            # Parent directory for all model results

def sanitise_model_name(name):
    """Replace any character that is not alphanumeric or underscore with underscore."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def save_results(results_dir, results_data, summary):
    """Save detailed results and summary to JSON files inside the results directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    details_file = os.path.join(results_dir, f"details_{timestamp}.json")
    summary_file = os.path.join(results_dir, f"summary_{timestamp}.json")

    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Saved] Detailed results → {details_file}")
    print(f"[Saved] Summary → {summary_file}")

def evaluate_model():
    print(f"--- ORAN-Bench Evaluation Start ---")
    print(f"Model: {MODEL_NAME}")
    print(f"API URL: {API_BASE_URL}")
    print(f"File:  {DATASET_PATH}")

    # Initialize OpenAI Client for LM Studio
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        # Test connection
        client.models.list()
    except Exception as e:
        print(f"\n[Error] Failed to connect to LM Studio server at {API_BASE_URL}")
        print(f"Details: {e}")
        print("Please ensure LM Studio Local Server is running.")
        return

    # Sanitise model name and create results folder
    safe_model = sanitise_model_name(MODEL_NAME)
    results_dir = os.path.join(RESULTS_BASE_DIR, safe_model)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}\n")

    # 1. Load dataset (JSON Lines format)
    dataset = []
    if not os.path.exists(DATASET_PATH):
        print(f"Error: File '{DATASET_PATH}' not found.")
        return

    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        dataset.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip malformed lines (e.g., empty brackets)
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
    results_data = []          # store details for each question

    start_time = time.time()

    # 2. Loop through questions (with interrupt handling)
    try:
        for i, item in enumerate(dataset):
            # Validate item structure
            if not isinstance(item, list) or len(item) < 3:
                print(f"Skipping malformed item at index {i}: {item}")
                continue

            question_text = item[0]
            options = item[1]
            correct_answer_index = str(item[2]).strip()

            # Build prompt
            prompt = (
                f"Context: Open Radio Access Network (O-RAN) Technical Specification.\n"
                f"Question: {question_text}\n"
                f"Options:\n"
            )
            for idx, opt in enumerate(options):
                prompt += f"{idx + 1}. {opt}\n"
            prompt += "\nTask: Provide ONLY the number (1, 2, 3, or 4) of the correct option. Do not explain."

            # Call LM Studio (via OpenAI Compatible API)
            try:
                # LM Studio uses the Chat Completion endpoint
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,  # Set to 0 for deterministic benchmarking
                    max_tokens=10   # We only expect a single digit
                )
                
                raw_output = response.choices[0].message.content.strip()

                # Extract the first digit found
                match = re.search(r'\d', raw_output)
                predicted_index = match.group() if match else None

                is_correct = (predicted_index == correct_answer_index)
                if is_correct:
                    correct_count += 1

                # Store result
                result_entry = {
                    "question_index": i,
                    "question": question_text,
                    "options": options,
                    "correct_index": correct_answer_index,
                    "model_raw_output": raw_output,
                    "predicted_index": predicted_index,
                    "is_correct": is_correct,
                    "error": None
                }

            except Exception as e:
                # On error, count as incorrect
                is_correct = False
                result_entry = {
                    "question_index": i,
                    "question": question_text,
                    "options": options,
                    "correct_index": correct_answer_index,
                    "model_raw_output": None,
                    "predicted_index": None,
                    "is_correct": False,
                    "error": str(e)
                }
                print(f"Error on question {i+1}: {e}")

            processed_count += 1
            results_data.append(result_entry)

            # Print individual result
            status = "✓" if is_correct else f"✗ (Model said {predicted_index}, Correct was {correct_answer_index})"
            print(f"[{processed_count}/{total_to_run}] {status}")

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user. Saving results so far...")

    # 3. Final (or partial) summary
    elapsed = time.time() - start_time
    accuracy = (correct_count / processed_count * 100) if processed_count > 0 else 0.0

    summary = {
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "dataset": DATASET_PATH,
        "total_questions_in_file": total_available,
        "questions_processed": processed_count,
        "correct_answers": correct_count,
        "accuracy_percent": round(accuracy, 2),
        "elapsed_seconds": round(elapsed, 2),
        "interrupted": processed_count < total_to_run
    }

    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Total Evaluated: {processed_count}")
    print(f"Correct:          {correct_count}")
    print(f"Accuracy:         {accuracy:.2f}%")
    print(f"Time:             {elapsed:.2f} s")
    print("="*30)

    # 4. Save everything
    save_results(results_dir, results_data, summary)

if __name__ == "__main__":
    evaluate_model()