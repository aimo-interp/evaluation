import argparse
import json
import random
from solutions.solve_a1d40b import solve_a1d40b

def main():
    parser = argparse.ArgumentParser(description="Generate a1d40b Augmented Dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="a1d40b_dataset.jsonl", help="Output file path (JSONL)")
    parser.add_argument("--max_val", type=int, default=10_000_000_000, help="Max value of exponent to solve")
    args = parser.parse_args()
    
    print(f"Starting generation of {args.num_samples} a1d40b samples...")
    print(f"Dataset will be saved to {args.output}")
    print("-" * 50)
    
    with open(args.output, "w", encoding="utf-8") as f:
        # Original problem
        orig_exp = 101
        result = solve_a1d40b(orig_exp)
        result["params"] = [orig_exp]
        result["is_original"] = True
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Sample 0 (Original) | limit_exponent={orig_exp} | Result={result['numeric_solution']}")

        # We sample limit_exponent in a range that keeps the problem mathematically consistent
        # Original was 101. Let's sample in [50, 500].
        exponents = random.sample(range(11, args.max_val), min(args.num_samples, 451))
        
        for idx, exp in enumerate(exponents):
            result = solve_a1d40b(exp)
            result["params"] = [exp]
            print(f"Sample {idx+1}/{args.num_samples} | limit_exponent={exp} | Result={result['numeric_solution']}")
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
    print("-" * 50)
    print(f"Successfully generated {len(exponents)} samples and saved to {args.output}")

if __name__ == '__main__':
    main()
