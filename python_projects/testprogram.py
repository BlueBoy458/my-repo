from quadratic_pro import solve

n = solve("15x**2+8x-7")
print(n)
print(type(n["x1"]))
if isinstance(n, dict):
    print("n is dict")
