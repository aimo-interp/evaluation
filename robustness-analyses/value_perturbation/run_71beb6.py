import argparse
import json
import random
from solutions.solve_71beb6 import solve_71beb6

def main():
    parser = argparse.ArgumentParser(description="Generate 71beb6 Augmented Dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="71beb6_dataset.jsonl", help="Output file path (JSONL)")
    args = parser.parse_args()
    
    print(f"Starting generation of {args.num_samples} 71beb6 samples...")
    print(f"Dataset will be saved to {args.output}")
    print("-" * 50)
    
    with open(args.output, "w", encoding="utf-8") as f:
        # Range of E where it's a power of 10. For the problem logic to hold as in solution,
        # we stick to 10^k. Original E=100 (k=2).
        # We can try E in {10, 100, 1000}.
        exponents = [10, 100, 1000]
        
        for idx, E in enumerate(exponents[:args.num_samples]):
            result = solve_71beb6(E)
            print(f"Sample {idx+1}/{args.num_samples} | E={E} | Result={result['numeric_solution']}")
            result["params"] = [E]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
    print("-" * 50)
    print(f"Successfully generated {min(len(exponents), args.num_samples)} samples and saved to {args.output}")

if __name__ == '__main__':
    main()
