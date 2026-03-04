import argparse
import os
import subprocess
import sys
import json

# Define the tasks based on their 6-character ID hashes
TASK_IDS = [
    {"id": "057f8a", "name": "dodola"},
    {"id": "1acac0", "name": "triangle"},
    {"id": "bbd91e", "name": "alice_board"},
    {"id": "a1d40b", "name": "fibonacci_prime"},
    {"id": "480182", "name": "triangle_segment"},
    {"id": "349493", "name": "delightful_seq"},
    {"id": "88c219", "name": "artificial_int"},
    {"id": "71beb6", "name": "digit_sum"},
    {"id": "1fce4b", "name": "three_digit_divisor"},
]

def map_task_to_dataset(tid):
    """Maps task ID to the standardized dataset filename in the datasets/ directory."""
    mapping = {
        "057f8a": "057f8a_dataset.jsonl",
        "1acac0": "1acac0_dataset.jsonl",
        "bbd91e": "bbd91e_dataset.jsonl",
        "a1d40b": "a1d40b_dataset.jsonl",
        "480182": "480182_dataset.jsonl",
        "349493": "349493_dataset.jsonl",
        "88c219": "88c219_dataset.jsonl",
        "71beb6": "71beb6_dataset.jsonl",
        "1fce4b": "1fce4b_dataset.jsonl",
    }
    return f"datasets/{mapping.get(tid, f'{tid}_dataset.jsonl')}"

def run_command(cmd, env_update=None):
    """Executes a shell command with an optional environment update and returns success."""
    print(f"Executing: {' '.join(cmd)}")
    
    env = os.environ.copy()
    if env_update:
        env.update(env_update)
    
    # Ensure the solutions directory is in PYTHONPATH so run scripts can import solve modules
    solutions_dir = os.path.abspath("solutions")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{solutions_dir}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = solutions_dir

    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Generate and Evaluate all IMO-symbolic tasks.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate per task.")
    parser.add_argument("--provider", type=str, choices=["openai", "gemini"], required=True, help="LLM Provider.")
    parser.add_argument("--model", type=str, default="", help="Specific model name or deployment name.")
    parser.add_argument("--delay", type=int, default=5, help="Delay between API requests.")
    parser.add_argument("--skip_gen", action="store_true", help="Skip the dataset generation phase.")
    args = parser.parse_args()

    # Ensure required directories exist
    for d in ["datasets", "results", "solutions"]:
        if not os.path.exists(d):
            os.makedirs(d)

    summary = {}

    print("=== IMO-Symbolic Pipeline Start ===")

    for task in TASK_IDS:
        tid = task["id"]
        tname = task["name"]
        
        run_script = f"run_{tid}.py"
        dataset_path = map_task_to_dataset(tid)
        result_path = f"results/{tid}_eval_results.jsonl"

        print(f"\n--- Task: {tname} ({tid}) ---")

        # Phase 1: Generation
        if not args.skip_gen:
            if os.path.exists(run_script):
                gen_cmd = [sys.executable, run_script, "--num_samples", str(args.num_samples), "--output", dataset_path]
                if not run_command(gen_cmd):
                    print(f"Error: Generation failed for {tname}. Skipping evaluation.")
                    continue
            else:
                print(f"Warning: Run script {run_script} not found. Checking if dataset exists...")

        # Phase 2: Evaluation
        if os.path.exists(dataset_path):
            eval_cmd = [
                sys.executable, "evaluate_models.py",
                "--dataset", dataset_path,
                "--output", result_path,
                "--provider", args.provider,
                "--delay", str(args.delay)
            ]
            if args.model:
                eval_cmd += ["--model", args.model]

            if run_command(eval_cmd):
                # Calculate Accuracy for the summary
                try:
                    correct = 0
                    total = 0
                    with open(result_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            total += 1
                            if data.get("is_correct"):
                                correct += 1
                    
                    acc = (correct / total * 100) if total > 0 else 0
                    summary[tname] = {"acc": acc, "count": f"{correct}/{total}"}
                except Exception as e:
                    print(f"Error parsing results for {tname}: {e}")
            else:
                print(f"Error: Evaluation failed for {tname}.")
        else:
            print(f"Error: Dataset {dataset_path} not found for {tname}.")

    # Final Report
    print("\n" + "="*50)
    print(f"{'TASK NAME':<25} | {'ACCURACY':<10} | {'SCORE'}")
    print("-" * 50)
    for name, stats in summary.items():
        print(f"{name:<25} | {stats['acc']:>8.2f}% | {stats['count']}")
    print("="*50)

if __name__ == "__main__":
    main()
