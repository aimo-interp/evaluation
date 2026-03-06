import math
from typing import Dict, Any

def generate_71beb6_text(N_exponent: int) -> str:
    """Generates the English text representation of the digit sum problem."""
    return (f"For a positive integer $n$, let $S(n)$ denote the sum of the digits of $n$ in base 10. "
            f"Compute $S(S(1)+S(2)+\\cdots+S(N))$ with $N=10^{{{N_exponent}}}-2$.")

def solution(E=100):
    # 1. Total sum to 10^E - 1 is mathematically base_prefix * 10^(E-1)
    base_prefix = 45 * E 
    zeros_count = E - 1 
    
    # 2. Number to subtract to get to 10^E - 2
    subtracted_value = 9 * E
    
    # Determine how many digits the subtracted value has
    D = int(math.floor(math.log10(subtracted_value))) + 1
    
    # Edge Case Guard: If E is very small (like E=1 or E=2), the zeros might not cover D.
    # This puzzle implies large E, but it's good practice to safeguard.
    if zeros_count < D:
        # Just use standard string conversion for tiny numbers to prevent negative nines_count
        return sum(int(c) for c in str(base_prefix * (10**zeros_count) - subtracted_value))

    # 3. Analyze the subtraction
    # Subtracting from the zeros borrows 1 from the base_prefix
    leading_part = base_prefix - 1
    leading_digits_sum = sum(int(d) for d in str(leading_part))
    
    # The zeros between the borrowed digit and the final D digits turn into 9s.
    nines_count = zeros_count - D
    middle_nines_sum = 9 * nines_count
    
    # The final D digits are calculated strictly mathematically.
    last_part = (10 ** D) - subtracted_value
    last_part_sum = sum(int(d) for d in str(last_part))
    
    # 4. The final result is the sum of these isolated components.
    final_S_X = leading_digits_sum + middle_nines_sum + last_part_sum
    
    return final_S_X

def solve_71beb6(E: int) -> Dict[str, Any]:
    """
    Solves for S( S(1) + S(2) + ... + S(N) ) where N = 10^E - 2.
    """
    
    return {
        "textual_problem": generate_71beb6_text(E),
        "numeric_solution": solution(E),
        "exponent": E,
        "params": [E]
    }

if __name__ == "__main__":
    result = solve_71beb6(100)
    print(f"Problem: {result['textual_problem']}")
    print(f"Solution: {result['numeric_solution']}")
