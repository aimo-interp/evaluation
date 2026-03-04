import argparse
import json
import random
import time
import multiprocessing
from solutions.solve_057f8a import solve_dodola_island

def worker(periods, return_dict):
    try:
        result = solve_dodola_island(periods)
        return_dict['result'] = result
    except Exception as e:
        return_dict['error'] = str(e)

def run_with_timeout(periods, timeout_seconds):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=worker, args=(periods, return_dict))
    p.start()
    p.join(timeout_seconds)
    if p.is_alive():
        p.terminate()
        p.join()
        return "TIMEOUT"
    if 'error' in return_dict:
        return f"ERROR: {return_dict['error']}"
    return return_dict.get('result')

def generate_periods():
    """
    Generates 3 periods in the range [10, 400] where each period is a multiple of 10.
    """
    # Sample 3 integers from {10, 20, 30, ..., 400}
    periods = [random.randint(1, 40) * 10 for _ in range(3)]
    return periods

def main():
    parser = argparse.ArgumentParser(description="Generate Dodola Island Augmented Dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="dodola_dataset.jsonl", help="Output file path (JSONL)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per calculation in seconds (default 5 min)")
    args = parser.parse_args()
    
    generated = 0
    
    print(f"Starting generation of {args.num_samples} samples...")
    print(f"Dataset will be saved to {args.output}")
    print(f"Timeout per calculation set to {args.timeout} seconds")
    print("-" * 50)
    
    with open(args.output, "w", encoding="utf-8") as f:
        while generated < args.num_samples:
            n = 3
            periods = generate_periods()
            print(f"Sample {generated + 1}/{args.num_samples} | N={n}, Periods={periods} | Computing...")
            
            start = time.time()
            result = run_with_timeout(periods, args.timeout)
            elapsed = time.time() - start
            
            if result == "TIMEOUT":
                print(f"  -> TIMEOUT after {args.timeout}s. Discarding and resampling.")
                continue
            elif isinstance(result, str) and result.startswith("ERROR"):
                print(f"  -> {result}. Discarding and resampling.")
                continue
            elif result is None or not isinstance(result, dict):
                print(f"  -> Received invalid result. Discarding and resampling.")
                continue
                
            print(f"  -> Success! Numeric solution = {result['numeric_solution']} (Took {elapsed:.2f}s)")
            
            record = {
                "textual_problem": result["textual_problem"],
                "numeric_solution": result["numeric_solution"],
                "periods": result["periods"],
                "params": result["periods"]
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            generated += 1
            
    print("-" * 50)
    print(f"Successfully generated {args.num_samples} samples and saved to {args.output}")

if __name__ == '__main__':
    main()