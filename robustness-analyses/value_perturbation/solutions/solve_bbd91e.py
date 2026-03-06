from typing import List, Dict, Any, Optional

def generate_alice_text(k: int, P: int, Q: int, modulo: int) -> str:
    """Generates the English text representation of the Alice Board problem."""
    # Note: n_min is typically k + 1. The original problem had n >= 11 for k=10.
    return (f"Alice writes all positive integers from 1 to n on the board for some positive integer n >= {k+1}. "
            f"Bob then erases {k} of them. The mean of the remaining numbers is {P}/{Q}. "
            f"The sum of the numbers Bob erased is S. What is the remainder when n * S is divided by {modulo}?")

def solve_erased_numbers(erased_count, mean_num, mean_den, modulo):
    # 1. Estimate n
    # The mean of the original sequence 1..n is (n + 1) / 2, which is roughly n/2.
    # Removing a small number of elements doesn't change the mean drastically.
    # Therefore, n / 2 is approximately mean_num / mean_den.
    # This implies: n ≈ 2 * mean_num / mean_den.

    # 2. Find the exact multiplier (k)
    # The remaining sum is (n - erased_count) * (mean_num / mean_den).
    # Since the sum must be an integer and the fraction is irreducible, 
    # the remaining count (n - erased_count) must be a multiple of mean_den.
    # Let's say: (n - erased_count) = k * mean_den.
    #
    # Substituting our approximation for n from step 1:
    # (2 * mean_num / mean_den) - erased_count ≈ k * mean_den
    # Multiplying by mean_den to isolate k algebraically:
    # k ≈ (2 * mean_num - erased_count * mean_den) / (mean_den^2)

    numerator_for_k = 2 * mean_num - erased_count * mean_den
    denominator_for_k = mean_den * mean_den

    # We find the exact integer k by rounding.
    # Adding half the denominator before integer division perfectly mirrors rounding 
    # to the nearest whole integer without using any floating-point approximations.
    k = (numerator_for_k + denominator_for_k // 2) // denominator_for_k

    # 3. Calculate exact initial values
    # Now we establish the exact total number of initial elements, n.
    n = k * mean_den + erased_count

    # Total sum of the original arithmetic progression 1..n is n*(n+1)/2.
    total_sum = n * (n + 1) // 2

    # 4. Calculate the sum of erased numbers (S)
    # The sum of the remaining elements is the remaining count multiplied by the mean.
    # Remaining count is (k * mean_den). Mean is (mean_num / mean_den).
    # Therefore, the remaining sum perfectly simplifies to k * mean_num.
    remaining_sum = k * mean_num

    # The sum of the erased numbers is simply the difference.
    S = total_sum - remaining_sum

    # 5. Find the final requested remainder
    result = (n * S) % modulo

    return result

def solve_alice_board(k: int, P: int, Q: int, modulo: int) -> Optional[Dict[str, Any]]:
    """
    Solves the parameterized Alice board problem.
    Returns a dictionary with the unique solution if found, else None.
    """


    return {
        "textual_problem": generate_alice_text(k, P, Q, modulo),
        "numeric_solution": solve_erased_numbers(k, P, Q, modulo),
        "modulo": modulo,
        "params": [modulo]
    }
    
if __name__ == "__main__":
    # Test with original parameters
    result = solve_alice_board(k=10, P=3000, Q=37, modulo=997)
    if result:
        print(f"Problem: {result['textual_problem']}")
        print(f"Solution: {result['numeric_solution']} (n={result['n']}, S={result['S']})")
    else:
        print("No unique solution found for default params.")
