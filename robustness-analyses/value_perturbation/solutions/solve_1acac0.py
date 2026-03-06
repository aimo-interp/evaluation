import math
from typing import Dict, Any

def generate_triangle_text(ab: int, r: int) -> str:
    """Generates the symbolic English text representation of the Triangle problem."""
    return f"Triangle $ABC$ has side length $AB = {ab}$ and circumradius $R = {r}$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?"

def solution(side_length, circumradius):
    """
    Calculates the greatest possible length of the altitude to a side of a triangle,
    given the length of that side and the circumradius.

    Parameters:
    side_length (float): The length of the side (chord of the circumcircle).
    circumradius (float): The radius of the circumcircle.

    Returns:
    float: The maximum possible length of the altitude to the given side.
    """

    # 1. Validate the inputs
    if side_length <= 0 or circumradius <= 0:
        raise ValueError("Side length and circumradius must be strictly positive.")

    if side_length > 2 * circumradius:
        raise ValueError("The side length cannot exceed the diameter of the circumcircle.")

    # 2. Calculate the distance from the circle's center to the chord
    half_side = side_length / 2.0
    distance_to_chord = math.sqrt(circumradius**2 - half_side**2)

    # 3. Calculate the maximum altitude
    # The maximum altitude is the distance from the center to the chord plus the radius
    max_altitude = circumradius + distance_to_chord

    return max_altitude

def solve_triangle(ab: int, r: int) -> Dict[str, Any]:
    """
    Solve the triangle problem for given parameters and return a structured dictionary.
    """
    textual_problem = generate_triangle_text(ab, r)
    numeric_solution = solution(ab, r)
    
    return {
        'textual_problem': textual_problem,
        'numeric_solution': numeric_solution,
        'ab': ab,
        'r': r,
        "params": [ab, r]
    }

if __name__ == "__main__":
    # Test with original parameters
    res = solve_triangle(ab=20, r=26)
    print("Test Output:")
    print(res["textual_problem"])
    print(f"Solution: {res['numeric_solution']}")