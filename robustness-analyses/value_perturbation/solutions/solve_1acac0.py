import math
from typing import Dict, Any

def generate_triangle_text(ab: int, r: int) -> str:
    """Generates the symbolic English text representation of the Triangle problem."""
    return f"Triangle $ABC$ has side length $AB = {ab}$ and circumradius $R = {r}$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?"

def calculate_max_altitude(ab: int, r: int) -> int:
    """Calculates the greatest possible length of altitude CD."""
    # Constraint Check
    if r < ab / 2:
        raise ValueError(f"Invalid Geometry: Circumradius {r} must be at least half of side AB ({ab/2}).")
    
    # Calculate the distance from the midpoint of AB to the circumcenter O
    # This uses the Pythagorean theorem: (AB/2)^2 + dist^2 = R^2
    half_ab = ab / 2
    dist_to_center = math.sqrt(r**2 - half_ab**2)
    
    # The maximum altitude CD occurs when C is on the far side of the circle.
    # CD = distance from chord to center + radius
    max_cd = dist_to_center + r
    
    # Returning as an integer assuming the dataset generation enforces Pythagorean triples
    return int(round(max_cd))

def solve_triangle(ab: int, r: int) -> Dict[str, Any]:
    """
    Solve the triangle problem for given parameters and return a structured dictionary.
    """
    textual_problem = generate_triangle_text(ab, r)
    numeric_solution = calculate_max_altitude(ab, r)
    
    return {
        'textual_problem': textual_problem,
        'numeric_solution': numeric_solution,
        'ab': ab,
        'r': r
    }

if __name__ == "__main__":
    # Test with original parameters
    res = solve_triangle(ab=20, r=26)
    print("Test Output:")
    print(res["textual_problem"])
    print(f"Solution: {res['numeric_solution']}")