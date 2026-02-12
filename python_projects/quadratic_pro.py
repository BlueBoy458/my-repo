from re import findall, Match, compile, split
from math import gcd
import cmath
import math
from typing import List, Literal, Self, Union
from collections.abc import Iterable
from functools import cached_property
from sqrtpy import PrettySqrt, PrettySqrtExpr, PrettyFraction
from time import sleep
# TO-DO LIST:
#   + Complete the quadratic equation logic for the Equation.solve() method (DONE)
#   + Simplify the equation and the result of the equation when it is displayed (DONE, probably)
#   + Add docstrings to each method to explain how it works.


class EquationError(Exception):
    """
    Raise when an equation is shown to be invalid.\n
    
    """

    pass

class InvalidAssignmentError(Exception):
    ...

class SymbolError(Exception):
    """ More than one symbol are used in an equation. """
    pass
    
def to_equation(x, sym="x"):
    if isinstance(x, Equation):
        return x
    elif isinstance(x, (str, int, float)):
        return Equation(x)
    elif isinstance(x, Iterable):
        return Equation.from_parse(x, symbolic=sym)

def to_number(x):
    return int(x) if float(x).is_integer() else float(x)

def sum_helper(n):
    result = sum([float(x) for x in n])
    return int(result) if result.is_integer() else result
    
def list_helper(x, t):
    for i in range(t):
        x.insert(0, 0)
def helper(x, y):
    
    diff = abs(len(x) - len(y))
    if len(x) > len(y):
        list_helper(y, diff) 
    elif len(x) < len(y):
        list_helper(x, diff) 
    return [x, y]

class Equation:
    """
    A class that solves simple symbolic equations.\n
    Currently supports any equation from degree 1 -> 3..
    Example:
    >>> a = Equation("a")
        print(a**2)
    a**2
    >>> print((a + 1)**2)
    a**2 + 2a + 1
    >>> print((a + 5) * (a - 3))
    a**2 + 2a - 15 = 0
    >>> x = Equation("x**2 -18x +7x**3")
        print(x**4)
    2401x**12 + 1372x**11 - 24402x**10 - 10556x**9 
    + 93745x**8 + 27144x**7 - 161352x**6 - 23328x**5 
    + 104976x**4 = 0
    """
    
    #Class variables (patterns)
    __number = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)"
    __symbols = r"\*?[a-zA-Z]"
    __exp_pattern = r"(?:\*\*\d+)"
    __exp1_pattern = compile(r"(\*\*0*1(?!\d))|\s")
    __zero_exp = compile(r"\*\*0*(?=\d)")
    __pattern = compile(r"(?<=[a-zA-Z]\*\*)\d+")
    
    
    def __init__(self, equation: str | Literal[0], allow_absurd: bool = True):
        self.unchanged = str(equation).strip()
        self.equation = Equation.__exp1_pattern.sub(r"", str(equation))
        self.equation = Equation.__zero_exp.sub(r"**", self.equation)
        
        if (
            self.equation.startswith("*")
            or any(self.equation.endswith(x) for x in "*+-")
                ):
            raise EquationError(
                f"Invalid equation: {self.equation} "
                f"(Unmodified: {self.unchanged}). \n"
                "(Did you add a leading/trailing asterisk or a trailing "
                "plus/minus operator?)"
                )
        
        elif (
            self.unchanged.startswith("=")
            or self.unchanged.endswith("=")
            ):
   
                raise InvalidAssignmentError(
                    "Nothing to assign after the assignment operator '=' (Did you forget to remove it?)"
                    f"\nExpression: {self.equation}"
                    )
        
        self.allow_absurd = allow_absurd
        simplify = self._quick_simplify()
        if hasattr(simplify, "equation"):
            self.equation = simplify.equation
        self.equation = compile(r'\+\+|--').sub('+', self.equation)
        self.equation = compile(r'\+-|-\+').sub('-', self.equation)
        self.equation = compile(r'\+\+|--').sub('+', self.equation)
            
    
    def __check_symbol(self, other):
        if self.symbol != other.symbol and not Equation.__isnumber(str(other)):
            if not(self.symbol or other.symbol):
                raise SymbolError(
                f"Unmatched symbol: '{self.symbol}', '{other.symbol}'"
                f"(From equation: {self}; {other})"
                )
    def __add__(self, other):
        other = to_equation(other).invert()
        self.__check_symbol(other)
        return Equation.from_parse(self.parse, other.parse, symbolic=self.symbol)
    
    
    def __radd__(self, other):
        return self.__add__(other)
     
     
    def __sub__(self, other):
        return self.__add__(Equation.invert(other))
        
        
    def __rsub__(self, other):
        return self.__sub__(other)
    
    
    def __mul__(self, other):
        if isinstance(other, (int, float)) or Equation.__isnumber(str(other)):
            n = [x * to_number(other) for x in self.parse]
            return Equation.from_parse(n, symbolic=self.symbol)
        elif isinstance(other, (str, Equation)):
            self.__check_symbol(other)
            if Equation.__isnumber(str(self)):
                return other.__mul__(to_number(str(self))) 
            other = to_equation(other)
            deg = self.degree + other.degree
            left, right = self.parse, other.parse
            result = {}
            for x in range(deg, -1, -1):
                result[x] = 0
            for ind, y in enumerate(left[::-1]):
                for i, z in enumerate(right[::-1]):
                    a = y * z
                    result[abs(deg - ind - i)] += a
            return Equation.from_parse(list(result.values())[::-1], symbolic=self.symbol)
        return NotImplemented
    
    
    def __rmul__(self, other):
        return self.__mul__(other)
        
        
    def __pow__(self, other):
        if not isinstance(other, int) or other < 0:
            if other < 0:
                raise TypeError(other)
            return NotImplemented
        result = 1
        while other:
            if other & 1:
                result = result * self
            self *= self
            other >>= 1
        return result
    
    
    #def zero_exp(self):

    def __divide_helper(self, other):
        res = self.parse[0] / other.parse[0]
        if self.degree - other.degree != 0:
            return Equation(f"{res}{self.symbol}**{self.degree - other.degree}")
        return Equation(res)
    def _divide(self, other, get_remainder = False):
        #numerator = self.as_dict()\
        #divisor = 0
        remainder = self
        #print(self.__divide_helper(other))
        a = ""
        while remainder.degree >= other.degree:
            divisor = remainder.__divide_helper(other)
            a += divisor
            remainder -= other * divisor 
        return a if not get_remainder else remainder
    def __truediv__(self, other):
        return self._divide(other)
    
    def __mod__(self, other):
        return self._divide(other, get_remainder=True)

        return 0
    def __pos__(self):
        return self
     
        
    def __neg__(self):
        return self.invert()
     
        
    def __bool__(self):
       return bool(self.find()[0])
       
       
    @cached_property
    def symbol(self) -> str:
        """
        Return the character of the symbol in the equation.
        """
        def __separate(n: list) -> str:
            return "'" + "','".join(n) + "'"
        
        pattern = compile(r"[a-zA-Z]")
        unallowed = compile(r"[^a-zA-Z0-9+\-*=()/.]")
        chars = pattern.findall(self.equation)
        chars_set = set(chars) 
        if len(chars_set) == 0:
            if self.equation == "0" or self.equation == 0 or self.allow_absurd:
                return ""
        i = len(chars_set)
        symbol_err = (
            f"Equation must have only one symbol, found {i}:"
                    f"{__separate(sorted(list(chars_set)))}\n (From expression '{self.equation}')"
            )
        if i > 1 or (i < 1 and not self.allow_absurd):
            raise SymbolError(symbol_err)
            
        elif special := unallowed.findall(self.equation):
            raise SymbolError(
                "Invalid special character(s) found in equation: '"
                + __separate(special)
            )
        else:
            return chars[0]
            
            
    @staticmethod
    def _get_degree(expr: str):
        # Find any matches that immediately precedes the equation symbol
        # with the double asterisks and does not have zero as coefficient
        exprs = " ".join(split(r"[+-]", expr))
        
        exp = Equation.__pattern.findall(exprs)
        if Equation.__isnumber(expr):
            return 1 if float(expr) == 0 else 0
        deg = max([int(x) for x in exp]) if exp else 1
        if deg == 0 and not Equation.__isnumber(expr):
            raise ZeroDivisionError(
                f"Equation cannot have an expression with zero degree" 
                f" (found {expr})"
                )
        return deg
        
        
    @cached_property
    def degree(self) -> int:
        """Return the largest degree of the equation."""
        #print(self.__find([0]))
        if not self.__find()[0]:
            return 0
        return Equation._get_degree(self.__find()[0])
        
        
    def __repr__(self):
        return self.__str__().replace(" = 0", "", 1)


    def __str__(self):
        exprs = self.find()
        if len(exprs) > 1 and not exprs[0] and Equation.__isnumber(exprs[1]):
            return str(exprs[1]).lstrip("+")
        if not any(exprs):
            return f"0{self.symbol}"
        if exprs[0].startswith("+"):
            exprs[0] = exprs[0].replace("+", "", 1)

        for i, x in enumerate(exprs):
            if i >= 1: 
                exprs[i] = x.replace("+", "+ ").replace("-", "- ")
        return f"{' '.join(exprs)}"
        
    def __float__(self):
        if Equation.__isnumber(str(self)):
            return float(str(self))
        return NotImplemented 
    
    def __int__(self):
        return int(float(self))
    
    def __cleanup(self, order, constant=""):
        sorted_order = []
        for x in order:
            if x in ["0", "+0", "-0"]:
                continue
            
            if x.lstrip("+-").startswith("0") and not Equation.__isnumber(x.lstrip("+-0")[0]):
                continue
            zero_prefix = x.removeprefix("0").removeprefix("+0").lstrip("0")
            if x.startswith("-0"):
                zero_prefix = "-" + x.lstrip("-0")
            sorted_order.append(zero_prefix)
        ops = "+-"
        for x, y in enumerate(sorted_order):
            if x == 0:
                continue
            if all(c != y[0] for c in ops):
                sorted_order[x] = "+" + y
        if sorted_order == []:
            sorted_order.append(constant)
        return sorted_order 
        
    
    def invert(self):
        eq = to_equation(self)
        expr = eq.parse
        return Equation.from_parse([-x for x in expr], symbolic=eq.symbol)
        
        
    @staticmethod
    def __simplify(expr, symbol):
        """
        A private helper. Attempts to simplify the equation by adding all symbolic terms with
        the same degtee together so there are only one symbolic term for each degree. 
        Example:
            5x**2+7x**2 => 12x**2
            -8x**3-8x+7x**2-12x+3x**3 -> -5x**3 + 7x**2 - 20x
        """
        coeff = compile(rf"{Equation.__number}(?={Equation.__symbols})(?=(\*\*\d+))?")
        if expr == []: 
            return [""]
        deg = Equation._get_degree(expr[0])
        res = []
        while deg > 0: 
            n = []
            for x in expr:
                if Equation._get_degree(x) == deg:
                    if coeff.search(x) is not None:
                        found = float(coeff.search(x).group(0))
                    else:
                        found = 1 if not x.startswith("-") else -1
                    n.append(int(found) if found.is_integer() else found)
            total = sum(n)
            t = total if total != 1 else ""
            e = f"{"-" if t == -1 else t}{symbol}"
            if deg > 1:
                res.append(f"{e}**{deg}")
            else:
                res.append(f"{e}")
            deg -= 1
        return res
    
    def __find(self, b_pattern=False, get_c=False, get_d=False):
    
        if b_pattern:
            return compile(rf"{self.__number}(?={self.__symbols})")
        sym_pattern = compile(rf"(?:{self.__number}|[+-])?{self.__symbols}{self.__exp_pattern}?")
        sym = sorted(
            sym_pattern.findall(self.equation), reverse=True, key=Equation._get_degree
        )
        if get_c:
            remainder = self.equation
            for term in sym:
                remainder = remainder.replace(term, "", 1)
            n = findall(Equation.__number, remainder)
            return str(sum_helper(n))
    
        return self.__cleanup(Equation.__simplify(sym, self.symbol))
        
        
    def find(self) -> List[str]:
        """Return a list containing all expressions of an equation,
        in sorted order. All elements of this list are strings."""
        self = to_equation(self)
        order = self.__find()
        
        constant = self.__find(get_c=True) 
        if not any(x in constant for x in "+-") and order:
            constant = "+" + constant

        if len(constant) > 1 and Equation.__isnumber(constant[1]) and int(constant[1]):  # Check if constant is provided
            order.append(constant)
        
        return order 
        
        
    def __isnumber(n):
            if isinstance(n, (int, float)):
                return True
            p = compile(r"\.|\s")
            try:
                return p.sub("r\1", n).lstrip("+-").isdigit()
            except TypeError:
                return False
    
    
    def is_valid(self, check_num = False, allow_absurd = False) -> bool:
        """ 
        Check if an equation is valid or not. An equation is considered to be valid
        if it satisfies all of the following: \n 
        — The equation should only have only one symbolic term. \n
        — The equation must not have any unsupported special characters. \n
        — If `allow_absurd` is False, the equation shall not be absurd.
        — Operators, especially '*' and '=', should be placed correctly in the equation. \n\n
        
        
        If the string is not qualified as a valid equation and `check_num` is True, 
        This method will check if the given string is a valid number.
        """
        is_equation = isinstance(self, Equation)
        if not check_num:
            if is_equation: 
                return bool(self.symbol)
            self = str(self).strip()
            try:
                n = Equation(self, allow_absurd)
                return bool(n.symbol)
            except (EquationError, ValueError):
                return False
            return True
        return Equation.__isnumber(str(self))
        
        
    @staticmethod
    def _from_parse(nums: List[int | float], sym_char="x") -> Self:
        helper = ""
        
        for ind, coeff in enumerate(nums):
            deg = len(nums) - (ind + 1)
            if deg > 0:
                c = ""
                if coeff > 0: 
                    c = coeff if coeff != 1 else ""
                else:
                    c = coeff if coeff != -1 else "-"
                op = "+" if not any(
                    str(coeff).startswith(x) 
                    for x in "+-"
                    ) else ""
                d = f"**{deg}" if deg != 1 else ""
                exponent = f"{op}{c}{sym_char}{d}" 
                helper += exponent
            elif deg == 0:
                helper += str(nums[ind])
        if helper == "0":
            helper += sym_char
        return Equation(helper) 
        
        
    @staticmethod
    def from_parse(
        nums: Union[int, float, Iterable[int | float]], 
        nums2: Iterable[int | float] | None = None,
        symbolic: str = "x"
        ) -> Self:
            if Equation.__isnumber(nums):
                nums = int(nums) if float(nums).is_integer() else float(nums)
            elif Equation.__isnumber(nums2):
                nums2 = int(nums2) if float(nums2).is_integer() else float(nums2)
            if isinstance(nums, (int, float)):
                nums = [nums]
            elif isinstance(nums2, (int, float)):
                nums2 = [nums2]
    
            eq = repr(Equation._from_parse(nums, sym_char=symbolic))
            if nums2: 
                eq2 = repr(Equation._from_parse(nums2, sym_char=symbolic))
                return Equation(eq + "=" + eq2)
            return Equation._from_parse(nums, sym_char=symbolic)
            
            
    def _quick_simplify(self):
        def convert(numbers):
            res = float(numbers)
            return res if not res.is_integer() else int(res)
        def helper(n, r):
            for x in range(r):
                n.insert(0, 0)
        if "=" not in self.equation:
            
            ...
        else:
            expr = self.equation.split("=")
            res = []
            try:
                left, right = [Equation(x).parse for x in expr]
            except EquationError: 
                if Equation.__isnumber(expr[0]):
                    right = Equation(expr[1]).parse
                    left = [0] * (len(right) - 1) + [convert(expr[0])]
                elif Equation.__isnumber(expr[1]):
                    left = Equation(expr[0]).parse
                    right = [0] * (len(left) - 1) + [convert(expr[1])]
                else:
                    raise InvalidAssignmentError(f"Invalid equation: \"{self.equation}\"")
                    
            except ValueError as e:
                raise InvalidAssignmentError(
                    f"Expected 1 assignment operator '=', found {len(expr)-1}\n"
                    f"(Original error message: {e})"
                    f"\nFrom expression {expr}"
                    f"\n(Unmodified: {self.unchanged})"
                    ) from e
                    
            L, R = len(left), len(right)
            diff = abs(L - R)
            if L < R:
                helper(left, diff)
            elif L > R:
                helper(right, diff)
            chars = set(findall(r"[a-zA-Z]", self.equation))
            sym = list(chars)[0] if chars else "x"
            for x in range(len(left)): 
                res.append(left[x] - right[x])
            return Equation.from_parse(res, symbolic = sym)
        
            
    @cached_property
    def parse(self) -> List[int | float]:
        """
        Parse the quadratic equation, and return a list containing all
        coefficients from the equation, if valid. The quadratic equation
        can be scrambled/unordered, such as 2x+8.5x**2+2. Also works well
        with cubic equations and n-degree equations (n >= 4).
        Example:

        """
        def get_operator(expr: str) -> Literal[-1, 1]:
            return -1 if expr.strip().startswith("-") else 1

        def f_value(expr: str) -> int | float:
            val = float(expr)
            return int(val) if val.is_integer() else val

        def get_value(expr: str, match_obj: Match):
            try:
                num = float(expr)
                return int(num) if num.is_integer() else num
            except ValueError:
                if match_obj:
                    return f_value(match_obj.group())
                return get_operator(expr)
                
        
        num = f_value(self.__find(get_c=True) or "0")
        sym = self.__find() 
        if not sym[0]:
            sym[0] = "0"
        if len(sym) < self.degree + 1:
            for x, y in enumerate(sym):
                y_degree = Equation._get_degree(y)
                if y_degree != self.degree - x:
                    sym.insert(x, 0)
            sym.extend([0] * (self.degree - len(sym)))
        for i in range(len(sym)):
            if isinstance(sym[i], str):
                sym[i] = get_value(sym[i], self.__find(b_pattern=True).search(sym[i]))
        sym.append(num)
        return sym
        
    
    def as_dict(self: "Equation", reverse=False, make_unique=False):

        n = self.parse[::-1]
        res = {}
        s = range(len(n)) if not reverse else range(len(n) - 1, -1, -1)
        for x in s:
            if make_unique and n[x] == 0:
                continue
            res[x] = n[x]
        
        return res
    
    @staticmethod
    def from_dict(self: dict):
        deg = sorted(list(self.keys()), reverse=True)
        a = []
        for x in range(max(deg), -1, -1):
            if x in deg:
                a.append(self[x])
            else:
                a.append(0)
        return Equation.from_parse(a)
    

    def assign(self, value):
        eq = to_equation(self)
        result = 0
        for i, x in enumerate(eq.parse[:-1]):
            result += x * (value)**(i + 1)
        return result + eq.parse[-1]
        
        
    def solve(
        self: Union[Self, str, int, Iterable[int | float]], 
        right_side: Union[str, int, Iterable[int | float]] = [], allow_imag=True,
    ) -> (PrettyFraction | dict[str, PrettySqrtExpr | int | PrettyFraction] | Literal["inf"] | None):
        """Return a dictionary containing all real roots and imaginary roots
        of an equation (if possible). \n
        If there are infinite possible solutions, return float("inf"). \n
        Otherwise, if there are no possible solutions, return None. \n
        All roots are simplified and are instances of the PrettySqrtExpr
        class (If the result contains a square root), which can be evaluated
        or used for calculating with other expressions. The single asterisk
        character ('*') can be used to imply multiplications between the
        coefficients and the symbol (optional).

        Example:
        ```python
            Equation("x**2-12x+3").solve()
            {'x1': 6 + sqrt(33), 'x2': 6 - sqrt(33)}
            Equation("x**2 - 5x + 7").solve()
            {'x1': 5/2 - 1/2*i*sqrt(3), 'x2': 5/2 + 1/2*i*sqrt(3)}
            Equation("12x-7+22x**2").solve() #Scrambled equation
            {'x1': -3/11 + 1/22*sqrt(190), 'x2': -3/11 - 1/22*sqrt(190)}
            Equation("5y**2-15y+12").solve() #Different symbol
            {'y1': 3/2 - 1/10*i*sqrt(15), 'y2': 3/2 + 1/10*i*sqrt(15)}
            ```
        """
        if Equation.__isnumber(right_side):
            right_side = [right_side]
        right_side = right_side or Equation(0)
        if isinstance(right_side, Iterable):
            right_side = Equation.from_parse(right_side) 
        
        if isinstance(self, Equation):
            equation = Equation.from_parse(self.parse, right_side.parse)
        elif isinstance(self, (int, str)):
            if Equation.is_valid(self, allow_absurd=True):
                equation = Equation.from_parse(Equation(self).parse, right_side.parse)
            elif Equation.is_valid(self, check_num=True) and right_side: 
                n = right_side.parse[-1] - self
                equation = Equation.from_parse(right_side.parse[0:-1] + [n]) 
                
        elif isinstance(self, Iterable):
            equation = Equation.from_parse(self)
            
        
        else:
            raise TypeError(
                "Expected 'int', 'float', 'str' or an iterable object, "
                f"but instances of {type(self).__name__} is not iterable")
        
        parsed_eq = equation.parse
        
        if parsed_eq == [0, 0]:
            return float("inf")

        if not equation.is_valid():    
            return None
        
        try:
            if (divisor := gcd(*parsed_eq)) > 1:
                # Attempt to simplify the coefficients by dividing each coefficient
                # by the greatest common divisor of a, b, c (if it exists)
                for x in range(len(parsed_eq)):
                    parsed_eq[x] //= divisor
        except TypeError:
            pass

        if equation.degree == 1:
            try:
                a, b = parsed_eq
                return PrettyFraction(-b, a)
            except ZeroDivisionError: 
                return None
        elif equation.degree == 2:
            a, b, c = parsed_eq
            delta = b**2 - 4 * a * c
            denominator = 2 * a
            solution = {f"{equation.symbol}1": None, f"{equation.symbol}2": None}

            def _safe_div(num, den):
                # Delegate to PrettySqrtExpr division when appropriate
                if isinstance(num, PrettySqrtExpr):
                    return num / den
                if isinstance(num, PrettyFraction):
                    res = num / PrettyFraction(den)
                    return int(res) if res.denominator == 1 else res
                if isinstance(num, int):
                    return num // den if num % den == 0 else PrettyFraction(num, den)
                # fallback
                return num / den

            if delta < 0 and not allow_imag:
                raise EquationError(
                    f"Equation {equation} has no real solutions \
                                    (allow_imag = False)"
                )
            if delta == 0:
                return {equation.symbol: _safe_div(-b, 2 * a)}

            else:
                delta_root = PrettySqrt(delta)
                n = [
                    _safe_div(-b + delta_root, denominator),
                    _safe_div(-b - delta_root, denominator),
                ]
                for x, y in enumerate(solution.keys()):
                    solution[y] = n[x]
            return solution

        elif equation.degree == 3:
            
            a, b, c, d = parsed_eq

            def cbrt_complex(z):
                if z == 0:
                    return 0
                return cmath.exp(cmath.log(z) / 3)

            A, B, C, D = a, b, c, d

            # Depressed cubic substitution x = y + shift
            shift = -B / (3 * A)
            p = (3 * A * C - B * B) / (3 * A * A)
            q = (2 * B**3 - 9 * A * B * C + 27 * A * A * D) / (27 * A**3)

            discr = (q / 2) ** 2 + (p / 3) ** 3

            roots = []

            def _simplify(z):
                # convert near-real complex numbers to real ints/floats
                if isinstance(z, complex):
                    if abs(z.imag) < 1e-12:
                        r = z.real
                        if abs(r - round(r)) < 1e-12:
                            return int(round(r))
                        return float(r)
                    return z
                return z

            if discr > 1e-14:
                # one real root, two complex
                sqrt_disc = cmath.sqrt(discr)
                u = cbrt_complex(-q / 2 + sqrt_disc)
                v = cbrt_complex(-q / 2 - sqrt_disc)
                y1 = u + v
                y2 = -(u + v) / 2 + (u - v) * cmath.sqrt(3) / 2 * 1j
                y3 = -(u + v) / 2 - (u - v) * cmath.sqrt(3) / 2 * 1j
                roots = [y1 + shift, y2 + shift, y3 + shift]
            elif abs(discr) <= 1e-14:
                # multiple roots (at least two equal)
                u = cbrt_complex(-q / 2)
                y1 = 2 * u
                y2 = -u
                roots = [y1 + shift, y2 + shift, y2 + shift]
            else:
                # three distinct real roots
                # use trigonometric solution
                rho = math.sqrt(-(p**3) / 27)
                # clamp value for acos
                val = (-q / 2) / rho
                val = max(min(val, 1), -1)
                phi = math.acos(val)
                m = 2 * math.sqrt(-p / 3)
                y1 = m * math.cos(phi / 3)
                y2 = m * math.cos((phi + 2 * math.pi) / 3)
                y3 = m * math.cos((phi + 4 * math.pi) / 3)
                roots = [y1 + shift, y2 + shift, y3 + shift]

            solution = {}
            for i, r in enumerate(roots, start=1):
                simplified = _simplify(r)
                if not allow_imag and isinstance(simplified, complex):
                    continue
                solution[f"{equation.symbol}{i}"] = _simplify(r)

            return solution
            

if __name__ == "__main__":
    n = Equation("2x**3-2x**2+8x=2x**2")
    print(n**2)