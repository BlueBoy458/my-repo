from sqrtpy import PrettyFraction, PrettySqrt
from re import compile 
from typing import List, Literal, Self, Union
from collections.abc import Iterable
from functools import cached_property

number = r"[+-]?(?:\d+\.\d*|\d+)"
exponent = r"(?:\*\*\d+)"
variables = compile(r"[a-zA-Z]")
remove_exp1 = compile(r"(\*\*0*1(?!\d))|\s")
whitespaces = compile(r"\s")
symbols = r"\*?[a-zA-Z]+"
no_num_exp = rf"{number}|[+-]?(?={symbols}{exponent})"
sym_pattern = compile(rf"(?:{number}|[+-])?{symbols}{exponent}?")
no_num_sym = compile(rf"(?={number}|[+-])?{symbols}{exponent}?")

class Polynomial:
    def __init__(self, equation):
        self.equation = whitespaces.sub("", equation)
        #print(sym_pattern.findall(self.equation))

    @cached_property
    def symbol(self):
        n = variables.findall(self.equation)
        return tuple(n)
    
    
    @cached_property
    def values(self):
        vars = [(no_num_sym.search(x).group(0), compile(number).search(x).group(0)) for x in sym_pattern.findall(self.equation)]
        return vars
    
    
    def __str__(self):
        return self.equation 
    
    def is_common(self, other):
        return True 
    

expr = Polynomial("x**2 + 8x + 7x - 2y - 2y= 3")
print(expr) 
print(expr.values)
