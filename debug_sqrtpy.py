from sqrtpy import PrettySqrt
x = PrettySqrt(16)
print('x terms:', x.terms, type(x))
print('x.as_number():', getattr(x,'as_number',None) and x.as_number(), type(x.as_number()))
print('x/2:', (x/2), type(x/2))
print('x/2 as repr:', repr(x/2))
print('(x/2).as_number():', getattr(x/2,'as_number',None) and x/2 if not isinstance(x/2,(int,float,complex)) else x/2)
