from typing import Dict, Any

def generate_a1d40b_text(limit_exponent: int) -> str:
    """Generates the English text representation of the Fibonacci prime factors problem."""
    return (f"The Fibonacci numbers are defined as follows: $F_0 = 0$, $F_1 = 1$, and $F_{{n+1}} = F_n + F_{{n-1}}$ for $n \\geq 1$. "
            f"There are $N$ positive integers $n$ strictly less than $10^{{{limit_exponent}}}$ such that $n^2 + (n+1)^2$ is a multiple of 5 but $F_{{n-1}}^2 + F_n^2$ is not. "
            f"How many prime factors does $N$ have, counted with multiplicity?")

def solution(limit_exponent):
    """
    Calculates the number of prime factors (with multiplicity) of N, 
    where N is the count of positive integers n < 10^limit_exponent 
    satisfying the given mathematical conditions.
    """
    
    # Based on the modular arithmetic derivation:
    # Condition A (modulo 5 polynomial) restricts n to 1 or 3 mod 5.
    # Condition B (Fibonacci identity + Pisano period for 5) eliminates 3 mod 5.
    # Thus, valid integers n strictly follow n ≡ 1 (mod 5).
    # The count N of such integers strictly less than 10^limit_exponent is exactly (10^limit_exponent) / 5.
    
    # We factor N mathematically: N = (10^limit_exponent) / 5
    # N = (2^limit_exponent * 5^limit_exponent) / 5
    # N = 2^limit_exponent * 5^(limit_exponent - 1)
    
    # The exponent of the prime factor 2 in the prime factorization of N
    exponent_of_base_2 = limit_exponent
    
    # The exponent of the prime factor 5 in the prime factorization of N
    exponent_of_base_5 = limit_exponent - 1
    
    # The total number of prime factors, counted with multiplicity, 
    # is the sum of the exponents in its prime factorization.
    total_prime_factors = exponent_of_base_2 + exponent_of_base_5
    
    return total_prime_factors

def solve_a1d40b(limit_exponent: int) -> Dict[str, Any]:
    """
    Calculates the number of prime factors (with multiplicity) of N.
    N = (10^limit_exponent) / 5 = 2^limit_exponent * 5^(limit_exponent - 1).
    Total factors = limit_exponent + (limit_exponent - 1) = 2*limit_exponent - 1.
    """
    
    return {
        "textual_problem": generate_a1d40b_text(limit_exponent),
        "numeric_solution": solution(limit_exponent),
        "limit_exponent": limit_exponent,
        "params": [limit_exponent]
    }

if __name__ == "__main__":
    # Test with original parameters
    result = solve_a1d40b(101)
    print(f"Problem: {result['textual_problem']}")
    print(f"Solution: {result['numeric_solution']}")
