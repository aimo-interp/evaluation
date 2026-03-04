import argparse
import json
import random
from solutions.solve_bbd91e import solve_alice_board

def main():
    parser = argparse.ArgumentParser(description="Generate Alice Board Augmented Dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="alice_board_dataset.jsonl", help="Output file path (JSONL)")
    args = parser.parse_args()
    
    generated = 0
    attempts = 0
    max_attempts = args.num_samples * 200 # Higher retry for uniqueness requirement
    
    print(f"Starting generation of {args.num_samples} Alice Board samples...")
    print(f"Dataset will be saved to {args.output}")
    print("-" * 50)
    
    with open(args.output, "w", encoding="utf-8") as f:
        while generated < args.num_samples and attempts < max_attempts:
            attempts += 1
            
            # Sampling parameters from ranges provided:
            # k: 5-30
            # P: 1000-10_000
            # Q: 20-100
            # modulo: 500-2000
            k = random.randint(5, 30)
            P = random.randint(1000, 10000)
            Q = random.randint(20, 100)
            modulo = random.randint(500, 2000)
            
            # Solve and check uniqueness
            result = solve_alice_board(k, P, Q, modulo)
            
            if result:
                # Success! Write to file
                record = {
                    "textual_problem": result["textual_problem"],
                    "numeric_solution": result["numeric_solution"],
                    "params": result["params"],
                    "n": result["n"],
                    "S": result["S"]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                generated += 1
                print(f"Sample {generated}/{args.num_samples} | k={k}, P/Q={P}/{Q}, n={result['n']}, S={result['S']} | Result={result['numeric_solution']}")
            
    print("-" * 50)
    if generated < args.num_samples:
        print(f"Only generated {generated} samples (Uniqueness constraint is strict).")
    else:
        print(f"Successfully generated {generated} samples and saved to {args.output}")

if __name__ == '__main__':
    main()
