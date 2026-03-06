from math import gcd
from functools import reduce
from typing import List, Dict, Any
import math
def generate_dodola_text(periods) -> str:
    """Generates the symbolic English text representation of the Dodola Island problem."""
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    text = f"Three airline companies operate flights from Dodola island. Each company has a different schedule of departures: "
    for i, p in enumerate(periods):
        nth = ordinals[i] if i < len(ordinals) else f"{i+1}th"
        text += f"The {nth} company departs every {p} days. "
    
    text += "What is the greatest positive integer d for which it is true that there will be d consecutive days without a flight from Dodola island, regardless of the departure times of the various airlines?"
    return text

def solution(periods) -> int:
    a,b,c = periods
    # Step 1: Find the total cycle length before the schedule repeats perfectly.
    # For 100, 120, and 150, the Least Common Multiple (LCM) is 600 days.
    cycle_length = math.lcm(a, b, c)
    
    # Step 2: In those 600 days, the 100-day airline flies 6 times.
    # This creates 6 "gaps" we need the other airlines to fill.
    total_gaps = cycle_length // a 
    
    # Step 3: The 150-day airline is stuck on a 50-day grid (GCD of 100 and 150).
    # It can only help fill 2 of those gaps (100 / 50 = 2).
    c_coverage = a // math.gcd(a, c)
    
    # Step 4: This forces the 120-day airline to cover the remaining 4 gaps (6 - 2 = 4).
    b_must_cover = total_gaps - c_coverage
    
    # Step 5: The 120-day airline shifts by 20 days at a time (GCD of 100 and 120).
    # To cover 4 gaps, it has to stretch its flights across a 60-day span (3 jumps of 20).
    b_shift = math.gcd(a, b)
    span_needed = (b_must_cover - 1) * b_shift
    
    # Step 6: If you center that 60-day stretch inside the 100-day gap, 
    # the maximum empty space left over bounds exactly to 80 days.
    max_gap = (a + span_needed) // 2 
    
    # If there are 80 days between two flights, there are 79 days WITHOUT a flight.
    return max_gap - 1

def solve_dodola_island(periods) -> Dict[str, Any]:
 
    textual_problem = generate_dodola_text(periods)
    min_max_gap = solution(periods)
    return {
        'textual_problem': textual_problem,
        'numeric_solution': min_max_gap,
        "params": periods
    }
