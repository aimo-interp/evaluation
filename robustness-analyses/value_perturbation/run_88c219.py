import argparse
import json
import random
from solutions.solve_88c219 import solve_88c219

def main():
    parser = argparse.ArgumentParser(description="Generate 88c219 Augmented Dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="88c219_dataset.jsonl", help="Output file path (JSONL)")
    args = parser.parse_args()
    
    print(f"Starting generation of {args.num_samples} 88c219 samples...")
    print(f"Dataset will be saved to {args.output}")
    print("-" * 50)
    
    with open(args.output, "w", encoding="utf-8") as f:
        # Original: (2, 40). We can augment upper_bound.
        upper_bounds = random.sample(range(30, 100), min(args.num_samples, 70))
        
        for idx, ub in enumerate(upper_bounds):
            result = solve_88c219(2, ub)
            print(f"Sample {idx+1}/{args.num_samples} | range=[2, {ub}] | Result={result['numeric_solution']}")
            result["params"] = [2, ub]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
    print("-" * 50)
    print(f"Successfully generated {len(upper_bounds)} samples and saved to {args.output}")

if __name__ == '__main__':
    main()
