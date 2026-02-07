from quadratic_pro import Equation, solve
import sys

n = input() or "8x**4 + 1 + 3x**2"

print(Equation(n).parse)
print(solve(n))
print(sys.version)