import argparse
import json
import random
import math
from solutions.solve_1acac0 import solve_triangle

def get_valid_integer_pairs_sampled(num_samples, min_val=10, max_val=1000000):
    """
    Randomly samples parameter pairs and keeps only those that yield integer solutions.
    Since we need R^2 = d^2 + (AB/2)^2, we can generate a Pythagorean triple 
    (a, b, c) such that AB = 2*a and R = c.
    
    Using Euclid's formula for generating Pythagorean triples:
    a = m^2 - n^2, b = 2mn, c = m^2 + n^2
    """
    valid_pairs = []
    attempts = 0
    max_attempts = num_samples * 1000  # Safety break
    
    while len(valid_pairs) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # We need c = m^2 + n^2 <= max_val
        # So m <= sqrt(max_val)
        limit = int(math.sqrt(max_val))
        if limit < 2:
            break
            
        m = random.randint(2, limit)
        n = random.randint(1, m - 1)
        
        # Generate primitive triple
        a = m**2 - n**2
        b = 2 * m * n
        c = m**2 + n**2
        
        # We can also use a multiplier k
        max_k = max_val // c
        if max_k < 1:
            continue
        k = random.randint(1, max_k)
        
        # Apply multiplier
        ka, kb, kc = k * a, k * b, k * c
        
        # We can assign AB/2 to either ka or kb
        for side in [ka, kb]:
            ab = side * 2
            r = kc
            if min_val <= ab <= max_val and min_val <= r <= max_val:
                valid_pairs.append((ab, r))
                if len(valid_pairs) >= num_samples:
                    break
                    
    return list(set(valid_pairs))

def main():
    parser = argparse.ArgumentParser(description="Generate Triangle Augmented Dataset")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="triangle_dataset.jsonl", help="Output file path (JSONL)")
    parser.add_argument("--min_val", type=int, default=10, help="Minimum parameter value")
    parser.add_argument("--max_val", type=int, default=10_000_000_000, help="Maximum parameter value")
    args = parser.parse_args()
    
    print(f"Sampling valid Pythagorean configurations for AB and R between {args.min_val} and {args.max_val}...")
    valid_pairs = get_valid_integer_pairs_sampled(args.num_samples, args.min_val, args.max_val)
    
    if not valid_pairs:
        print("No valid integer pairs found in the given range!")
        return
        
    print(f"Successfully sampled {len(valid_pairs)} mathematically valid configurations.")
    
    generated = 0
    print(f"Starting generation of {args.num_samples} samples...")
    print("-" * 50)
    
    with open(args.output, "w", encoding="utf-8") as f:
        # Original problem
        orig_ab, orig_r = 20, 26
        result = solve_triangle(orig_ab, orig_r)
        result["params"] = [orig_ab, orig_r]
        result["is_original"] = True
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Sample 0 (Original) | AB={orig_ab}, R={orig_r} | Solution: {result['numeric_solution']}")

        for ab, r in valid_pairs[:args.num_samples]:
            result = solve_triangle(ab, r)
            
            print(f"Sample {generated + 1}/{args.num_samples} | AB={ab}, R={r} | Solution: {result['numeric_solution']}")
            result["params"] = [result["ab"], result["r"]]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            generated += 1
            
    print("-" * 50)
    print(f"Successfully generated {generated} samples and saved to {args.output}")

if __name__ == '__main__':
    main()