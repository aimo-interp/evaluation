import argparse
import json
import random
from solutions.solve_71beb6 import solve_71beb6

def main():
    parser = argparse.ArgumentParser(description="Generate 71beb6 Augmented Dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="71beb6_dataset.jsonl", help="Output file path (JSONL)")
    parser.add_argument("--max_val", type=int, default=10_000_000_000, help="Max value of multiplier to solve")
    args = parser.parse_args()
    
    print(f"Starting generation of {args.num_samples} 71beb6 samples...")
    print(f"Dataset will be saved to {args.output}")
    print("-" * 50)
    
    with open(args.output, "w", encoding="utf-8") as f:
        # Original problem
        orig_E = 100
        result = solve_71beb6(orig_E)
        result["params"] = [orig_E]
        result["is_original"] = True
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Sample 0 (Original) | E={orig_E} | Result={result['numeric_solution']}")

        
        for idx in range(args.num_samples):
            E = random.randint(11, args.max_val)
            result = solve_71beb6(E)
            print(f"Sample {idx+1}/{args.num_samples} | E={E} | Result={result['numeric_solution']}")
            result["params"] = [E]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
    print("-" * 50)
    print(f"Successfully generated {args.num_samples} samples and saved to {args.output}")

if __name__ == '__main__':
    main()
