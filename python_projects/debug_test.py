from quadratic_pro import solve
r = solve('x**2-4')
print('solve->', r)
print('type x1', type(r['x1']))
print('repr x1', repr(r['x1']))
print('type x1+3', type(r['x1']+3), 'repr', repr(r['x1']+3))
