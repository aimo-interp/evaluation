import argparse
import json
import random
from solutions.solve_480182 import solve_480182

def main():
    parser = argparse.ArgumentParser(description="Generate 480182 Augmented Dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="480182_dataset.jsonl", help="Output file path (JSONL)")
    parser.add_argument("--max_val", type=int, default=1_000, help="Max value of multiplier to solve")

    args = parser.parse_args()
    
    print(f"Starting generation of {args.num_samples} 480182 samples...")
    print(f"Dataset will be saved to {args.output}")
    print("-" * 50)
    
    generated = 0
    attempts = 0
    max_attempts = args.num_samples * 1000000
    
    with open(args.output, "w", encoding="utf-8") as f:
        # Original problem
        orig_bc, orig_ca, orig_ab = 108, 126, 39
        result = solve_480182(orig_bc, orig_ca, orig_ab)
        result["params"] = [orig_bc, orig_ca, orig_ab]
        result["is_original"] = True
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Sample 0 (Original) | a={orig_bc}, b={orig_ca}, c={orig_ab} | Result={result['numeric_solution']}")

        while generated < args.num_samples and attempts < max_attempts:
            attempts += 1
        
            # 1. Pick 'a' first (must be large enough to allow c >= 10 and a > 2c)
            # Minimum 'a' is 21.
            a = random.randint(100, args.max_val - 5) 

            # 2. Pick 'c' based on your constraint: 10 <= c < a/2
            c = random.randint(10, (a // 2) - 1)

            # 3. Pick 'b' to satisfy BOTH:
            # Triangle Inequality: b < a + c
            # Your Constraint: b > a
            # So: a < b < a + c
            lower_b = a + 1
            upper_b = a + c - 1
            
            if lower_b > upper_b:
                # This happens if c is too small to provide any integer space for b
                continue
            
            b = random.randint(lower_b, upper_b)

            result = solve_480182(a, b, c)
            # Result could be complex, let's just make sure m+n isn't absurdly large
                
            print(f"Sample {generated + 1}/{args.num_samples} | a={a}, b={b}, c={c} | Result={result['numeric_solution']}")
            result["params"] = [a, b, c]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            generated += 1
            
    print("-" * 50)
    print(f"Successfully generated {generated} samples and saved to {args.output}")

if __name__ == '__main__':
    main()
