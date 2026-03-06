from fractions import Fraction
from typing import Dict, Any

def generate_480182_text(bc: int, ca: int, ab: int) -> str:
    """Generates the English text representation of the Triangle segment problem."""
    return (f"Let $ABC$ be a triangle with $BC={bc}$, $CA={ca}$, and $AB={ab}$. "
            f"Point $X$ lies on segment $AC$ such that $BX$ bisects $\\angle CBA$. "
            f"Let $\\omega$ be the circumcircle of triangle $ABX$. "
            f"Let $Y$ be a point on $\\omega$ different from $X$ such that $CX=CY$. "
            f"Line $XY$ meets $BC$ at $E$. The length of the segment $BE$ can be written as $\\frac{{m}}{{n}}$, "
            f"where $m$ and $n$ are coprime positive integers. Find $m+n$.")


def solution(bc: int, ca: int, ab: int) -> int:
    """
    Finds the length of segment BE as a fraction m/n and returns m + n.

    Args:
        bc: Length of side BC (a)
        ca: Length of side CA (b)
        ab: Length of side AB (c)

    Returns:
        The sum of the numerator (m) and denominator (n) of the simplified fraction for BE.
    """
    # Use Fraction to maintain exact rational arithmetic
    a = Fraction(bc)
    b = Fraction(ca)
    c = Fraction(ab)

    # Step 1: Find CX using the Angle Bisector Theorem
    cx = (a * b) / (a + c)

    # Step 2: Power of point C with respect to circumcircle of ABX
    # Power_C = CX * CA
    power_c = cx * b

    # Step 3: Find the second intersection of the circumcircle with line BC
    # The circumcircle intersects line BC at B and another point B'.
    # Power_C is also equal to CB * CB' (which is a * CB').
    cb_prime = power_c / a

    # Step 4: Find the coordinate of E on line BC
    # Let C be the origin (0) and B be at coordinate 'a'.
    # Line XY is the radical axis of the circumcircle and circle C (radius CX).
    # E has the same power with respect to both circles.
    # Power of E w.r.t. circumcircle = (x - a)(x - cb_prime)
    # Power of E w.r.t. circle C = x^2 - cx^2
    # Equating them: x^2 - x(a + cb_prime) + a * cb_prime = x^2 - cx^2
    # Solving for x: x(a + cb_prime) = a * cb_prime + cx^2

    x = (a * cb_prime + cx**2) / (a + cb_prime)

    # Step 5: Calculate distance BE
    be = abs(a - x)

    # Step 6: Return m + n (Fraction automatically reduces to coprime integers)
    return be.numerator + be.denominator

def solve_480182(bc: int, ca: int, ab: int) -> Dict[str, Any]:
    """
    Finds the length of segment BE as a fraction m/n and returns m + n.
    Args:
        bc: Length of side BC (a)
        ca: Length of side CA (b)
        ab: Length of side AB (c)
    Returns:
        The sum of the numerator (m) and denominator (n).
    """

    
    return {
        "textual_problem": generate_480182_text(bc, ca, ab),
        "numeric_solution": solution(bc, ca, ab),
        "params": [bc, ca, ab]
    }

if __name__ == "__main__":
    # Test with original parameters
    result = solve_480182(108, 126, 39)
    print(f"Problem: {result['textual_problem']}")
    print(f"Solution (m+n): {result['numeric_solution']}")
