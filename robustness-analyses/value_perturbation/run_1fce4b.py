import argparse
import json
import random
from solutions.solve_1fce4b import solve_1fce4b

def main():
    parser = argparse.ArgumentParser(description="Generate 1fce4b Augmented Dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="1fce4b_dataset.jsonl", help="Output file path (JSONL)")
    args = parser.parse_args()
    
    print(f"Starting generation of {args.num_samples} 1fce4b samples...")
    print(f"Dataset will be saved to {args.output}")
    print("-" * 50)
    
    with open(args.output, "w", encoding="utf-8") as f:
        # exponent can be anything large. We can sample in [1000, 10000].
        exponents = random.sample(range(1000, 10001), min(args.num_samples, 9001))
        
        for idx, exp in enumerate(exponents):
            result = solve_1fce4b(exp)
            print(f"Sample {idx+1}/{args.num_samples} | exponent={exp} | Result={result['numeric_solution']}")
            result["params"] = [exp]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
    print("-" * 50)
    print(f"Successfully generated {len(exponents)} samples and saved to {args.output}")

if __name__ == '__main__':
    main()
