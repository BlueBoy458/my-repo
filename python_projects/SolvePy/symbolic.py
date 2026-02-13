from sqrtpy import PrettyFraction, PrettySqrt
from re import compile 
from math import prod
from typing import List, Literal, Self, Union
from collections.abc import Iterable
from collections import defaultdict
from functools import cached_property

number = r"[+-]?(?:\d+\.\d*|\d+)"
num = r"(?:\d+\.\d*|\d+)"
exponent = compile(r"(?:\*\*\d+)")
exp = compile(r"(?:\*\*)?(\d+)")
leading_zero_exp = compile(r"\*\*0*(?=\d)")

variables = compile(r"[a-zA-Z]")
remove_exp1 = compile(r"(\*\*0*1(?!\d))|\s")
whitespaces = compile(r"\s")
symbols = r"\*?[a-zA-Z]+"

no_num_exp = compile(rf"{number}|[+-]?(?={symbols}{exponent.pattern})")
exact_zero_exp = compile(rf"(?:{num}{symbols}|{num}|{symbols})\*\*0\b")
sym_pattern = compile(rf"(?:{number}|[+-])?{symbols}{exponent.pattern}?")
sym_pattern_mul = compile(rf"{sym_pattern.pattern}(?:\*(?:{num})?(?:{symbols})?)*")
no_num_sym = compile(rf"(?={number}|[+-])?{symbols}{exponent.pattern}?")
#print(str(sym_pattern))
constants = compile(rf"(?<!\*\*){number}(?!{symbols}{exponent.pattern}?)")

def sum_helper(n):
    result = sum([float(x) for x in n])
    return int(result) if result.is_integer() else result

def to_number(result):
    try:
        return int(result) if float(result).is_integer() else float(result)
    except ValueError:
        return 1

def is_number(n):
    try:
        float(n)
        return True
    except ValueError:
        return False
    

def to_poly(expr):
    return Polynomial(expr)

class Polynomial:
    def __init__(self, equation):
        self.equation = whitespaces.sub("", equation)
        self.equation = remove_exp1.sub("", self.equation)
        self.equation = exact_zero_exp.sub("1", self.equation)

    @staticmethod 
    def _symbol(expr):
        return tuple(variables.findall(expr))
    

    @cached_property
    def symbol(self):
        return Polynomial._symbol(self.equation)
    
    @staticmethod 
    def _degree(expr):
        if is_number(expr):
            return "0"
        if exponent.search(expr):
            return str(max([x for x in exp.findall(expr)]))
        return "0"
    
    @cached_property
    def degree(self):
        return int(Polynomial._degree(self.equation))
    

    @cached_property
    def values(self):
        
        #vars = [(no_num_sym.search(x).group(0), to_number(no_num_exp.search(x).group(0))) 
                #for x in sym_pattern.findall(self.equation)]
        # res = defaultdict(int)
        # for x, y in vars:
        #     res[x] += y 

        # constant = sum_helper(constants.findall(self.equation))
        # res[0] = constant
        # return dict(res)
        return sorted(sym_pattern.findall(self.equation), key = Polynomial._degree, reverse=True)
    
    
    def __str__(self):
        return self.equation 
    
    def is_common(self, other):
        return to_poly(self).symbol == to_poly(other).symbol
    
    @staticmethod
    def _multiply(eq):
        def helper(expr, sym):
            deg = 0
            for x in expr:
                if Polynomial(x).symbol
            ...

        n = sym_pattern.findall(eq)
        all_symbols = compile(symbols).findall(eq)
        same_symbol = [x for x in all_symbols if all_symbols.count(x) > 1]
        res = prod([to_number(no_num_exp.search(x).group(0)) for x in n])

        return helper(n, "y") #same_symbol
        ...    

#expr = Polynomial("xy**2 + xy**2 + 8x + 7x - 2y - 2y - 3 - 4")
#expr = Polynomial("2x**3 - 3x**4*3y**2*3y")
n = "3x**4*3y**2*3y*3"
print(Polynomial(n).degree)
print(Polynomial(n).values)
print(Polynomial._multiply(n))
