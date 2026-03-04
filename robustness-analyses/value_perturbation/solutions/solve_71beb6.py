import math
from typing import Dict, Any

def generate_71beb6_text(N_exponent: int) -> str:
    """Generates the English text representation of the digit sum problem."""
    return (f"For a positive integer $n$, let $S(n)$ denote the sum of the digits of $n$ in base 10. "
            f"Compute $S(S(1)+S(2)+\\cdots+S(N))$ with $N=10^{{{N_exponent}}}-2$.")

def solve_71beb6(E: int) -> Dict[str, Any]:
    """
    Solves for S( S(1) + S(2) + ... + S(N) ) where N = 10^E - 2.
    """
    k = int(math.log10(E))
    zeros_count = E + k - 1
    subtracted_value = 9 * E
    leading_digits_sum = 4 + 4
    D = int(math.floor(math.log10(subtracted_value))) + 1
    nines_count = zeros_count - D
    middle_nines_sum = 9 * nines_count
    last_part = (10 ** D) - subtracted_value
    digit_1 = (last_part // 100) % 10
    digit_2 = (last_part // 10) % 10
    digit_3 = last_part % 10
    last_part_sum = digit_1 + digit_2 + digit_3
    final_S_X = leading_digits_sum + middle_nines_sum + last_part_sum
    
    return {
        "textual_problem": generate_71beb6_text(E),
        "numeric_solution": final_S_X,
        "exponent": E
    }

if __name__ == "__main__":
    result = solve_71beb6(100)
    print(f"Problem: {result['textual_problem']}")
    print(f"Solution: {result['numeric_solution']}")
