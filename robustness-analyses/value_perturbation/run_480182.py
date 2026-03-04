import argparse
import json
import random
from solutions.solve_480182 import solve_480182

def main():
    parser = argparse.ArgumentParser(description="Generate 480182 Augmented Dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="480182_dataset.jsonl", help="Output file path (JSONL)")
    args = parser.parse_args()
    
    print(f"Starting generation of {args.num_samples} 480182 samples...")
    print(f"Dataset will be saved to {args.output}")
    print("-" * 50)
    
    generated = 0
    attempts = 0
    max_attempts = args.num_samples * 100
    
    with open(args.output, "w", encoding="utf-8") as f:
        while generated < args.num_samples and attempts < max_attempts:
            attempts += 1
            
            # Sampling side lengths BC=a, CA=b, AB=c.
            # Triangle inequality: a + b > c, a + c > b, b + c > a.
            # Ranges chosen around original (108, 126, 39).
            a = random.randint(50, 200)
            b = random.randint(50, 200)
            c = random.randint(20, a + b - 1)
            
            # Ensure triangle inequality is fully satisfied
            if not (a + b > c and a + c > b and b + c > a):
                continue
            
            result = solve_480182(a, b, c)
            # Result could be complex, let's just make sure m+n isn't absurdly large
            if result["numeric_solution"] > 10000:
                continue
                
            print(f"Sample {generated + 1}/{args.num_samples} | a={a}, b={b}, c={c} | Result={result['numeric_solution']}")
            result["params"] = [a, b, c]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            generated += 1
            
    print("-" * 50)
    print(f"Successfully generated {generated} samples and saved to {args.output}")

if __name__ == '__main__':
    main()
