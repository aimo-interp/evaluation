from typing import Dict, Any

def generate_88c219_text(lower_bound: int, upper_bound: int) -> str:
    """Generates the English text representation of the artificial integers problem."""
    return (f"For positive integers $x_1,\\ldots, x_n$ define $G(x_1, \\ldots, x_n)$ to be the sum of their $\\frac{{n(n-1)}}{{2}}$ pairwise greatest common divisors. "
            f"We say that an integer $n \\geq 2$ is \\emph{{artificial}} if there exist $n$ different positive integers $a_1, ..., a_n$ such that "
            f"\\[a_1 + \\cdots + a_n = G(a_1, \\ldots, a_n) +1.\\] Find the sum of all artificial integers $m$ in the range ${lower_bound} \\leq m \\leq {upper_bound}$.")

def solve_88c219(lower_bound: int, upper_bound: int) -> Dict[str, Any]:
    """
    Solves the artificial integers problem.
    Integers n >= 5 are artificial.
    """
    first_artificial = 5
    actual_start = max(lower_bound, first_artificial)
    actual_end = upper_bound
    number_of_terms = max(0, actual_end - actual_start + 1)
    final_sum = number_of_terms * (actual_start + actual_end) // 2
    
    return {
        "textual_problem": generate_88c219_text(lower_bound, upper_bound),
        "numeric_solution": final_sum,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }

if __name__ == "__main__":
    result = solve_88c219(2, 40)
    print(f"Problem: {result['textual_problem']}")
    print(f"Solution: {result['numeric_solution']}")
