from typing import List, Dict, Any, Optional

def generate_alice_text(k: int, P: int, Q: int, modulo: int) -> str:
    """Generates the English text representation of the Alice Board problem."""
    # Note: n_min is typically k + 1. The original problem had n >= 11 for k=10.
    return (f"Alice writes all positive integers from 1 to n on the board for some positive integer n >= {k+1}. "
            f"Bob then erases {k} of them. The mean of the remaining numbers is {P}/{Q}. "
            f"The sum of the numbers Bob erased is S. What is the remainder when n * S is divided by {modulo}?")

def solve_alice_board(k: int, P: int, Q: int, modulo: int) -> Optional[Dict[str, Any]]:
    """
    Solves the parameterized Alice board problem.
    Returns a dictionary with the unique solution if found, else None.
    """
    # approx_n calculation based on mean
    # (n+1)/2 approx P/Q => n approx 2P/Q - 1
    approx_n = int((2 * P) // Q)
    
    possible_solutions = []

    # Broad search range for n around the approximation
    for n in range(max(k + 1, approx_n - k - 10), approx_n + k + 20):
        # Total sum of 1...n
        total_sum = n * (n + 1) // 2
        
        # Bob erases k numbers. Remaining count is n - k.
        # Sum of remaining = Total_Sum - S
        # Mean = (Total_Sum - S) / (n - k) = P / Q
        # Q * (Total_Sum - S) = P * (n - k)
        # S = Total_Sum - (P * (n - k) / Q)
        
        num_q = (total_sum * Q) - (P * (n - k))
        
        if num_q % Q == 0:
            S = num_q // Q
            
            # S must be achievable by erasing k distinct numbers in [1, n]
            # Smallest S: sum of 1...k
            min_s = k * (k + 1) // 2
            # Largest S: sum of (n-k+1)...n
            max_s = k * (2 * n - k + 1) // 2
            
            if min_s <= S <= max_s:
                possible_solutions.append({
                    "n": n,
                    "S": S,
                    "remainder": (n * S) % modulo
                })

    # Requirement: The parameters must generate a UNIQUE solution
    if len(possible_solutions) == 1:
        sol = possible_solutions[0]
        return {
            "textual_problem": generate_alice_text(k, P, Q, modulo),
            "numeric_solution": sol["remainder"],
            "n": sol["n"],
            "S": sol["S"],
            "params": {"k": k, "P": P, "Q": Q, "modulo": modulo}
        }
    
    return None

if __name__ == "__main__":
    # Test with original parameters
    result = solve_alice_board(k=10, P=3000, Q=37, modulo=997)
    if result:
        print(f"Problem: {result['textual_problem']}")
        print(f"Solution: {result['numeric_solution']} (n={result['n']}, S={result['S']})")
    else:
        print("No unique solution found for default params.")
