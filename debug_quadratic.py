from quadratic_pro import Equation
q = Equation('x**2-4')
print('equation:', q.equation)
print('symbol:', q.symbol)
print('parse:', q.parse)
print([type(x) for x in q.parse])
print("Done")
