from typing import Dict, Any

def generate_a1d40b_text(limit_exponent: int) -> str:
    """Generates the English text representation of the Fibonacci prime factors problem."""
    return (f"The Fibonacci numbers are defined as follows: $F_0 = 0$, $F_1 = 1$, and $F_{{n+1}} = F_n + F_{{n-1}}$ for $n \\geq 1$. "
            f"There are $N$ positive integers $n$ strictly less than $10^{{{limit_exponent}}}$ such that $n^2 + (n+1)^2$ is a multiple of 5 but $F_{{n-1}}^2 + F_n^2$ is not. "
            f"How many prime factors does $N$ have, counted with multiplicity?")

def solve_a1d40b(limit_exponent: int) -> Dict[str, Any]:
    """
    Calculates the number of prime factors (with multiplicity) of N.
    N = (10^limit_exponent) / 5 = 2^limit_exponent * 5^(limit_exponent - 1).
    Total factors = limit_exponent + (limit_exponent - 1) = 2*limit_exponent - 1.
    """
    total_prime_factors = 2 * limit_exponent - 1
    
    return {
        "textual_problem": generate_a1d40b_text(limit_exponent),
        "numeric_solution": total_prime_factors,
        "limit_exponent": limit_exponent
    }

if __name__ == "__main__":
    # Test with original parameters
    result = solve_a1d40b(101)
    print(f"Problem: {result['textual_problem']}")
    print(f"Solution: {result['numeric_solution']}")
