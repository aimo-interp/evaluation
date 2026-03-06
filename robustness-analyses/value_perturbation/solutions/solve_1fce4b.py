from typing import Dict, Any

def generate_1fce4b_text(exponent, digits_in_x, k_difference) -> str:
    """Generates the English text representation of the three-digit number divisor problem."""
    return (f"Find the {digits_in_x}-digit number $n$ such that writing any other {digits_in_x}-digit number $10^{{{exponent}}}$ times in a row "
            f"and $10^{{{exponent}}}+{k_difference}$ times in a row results in two numbers divisible by $n$.")

def solution(digits_in_x, k_difference):
    # The base of our number system
    base = 10
    
    # Calculate the numerator of the invariant difference expression: 10^(3*2) - 1
    numerator = base ** (digits_in_x * k_difference) - 1
    
    # Calculate the denominator for the geometric series sum: 10^3 - 1
    denominator = base ** digits_in_x - 1
    
    # The invariant value that n must divide. For 3 digits and a difference of 2, this evaluates to 1001.
    invariant_difference = numerator // denominator
    
    # Since 1001 = 7 * 11 * 13, its only 3-digit divisor is 11 * 13 = 143.
    # We find this mathematically by isolating it from the smallest prime factor (7).
    # smallest_prime_factor = 7
    # n = invariant_difference // smallest_prime_factor

    from sympy import divisors

    n = list(filter(lambda x: 100 <= x < 1000, divisors(invariant_difference)))[0]
    
    return n

def solve_1fce4b(exponent, digits_in_x=3, k_difference=2) -> Dict[str, Any]:
    """
    Solves for n. n must divide the difference, which reduces to 1001 for any exponent since 10^3-1 
    divides 10^{3k}-1. For 3 digits, n = 143 (1001 / 7).
    """
    # The solution logic provided evaluates to 143 for any such problem structure.
    n = solution(digits_in_x, k_difference)
    
    return {
        "textual_problem": generate_1fce4b_text(exponent, digits_in_x, k_difference),
        "numeric_solution": n,
        "exponent": exponent,
        "digits_in_x": digits_in_x,
        "k_difference": k_difference,
        "params": [exponent, digits_in_x, k_difference]
    }

if __name__ == "__main__":
    result = solve_1fce4b(2024)
    print(f"Problem: {result['textual_problem']}")
    print(f"Solution: {result['numeric_solution']}")
