from re import match, findall, Match, sub
from math import gcd
import cmath
import math
from typing import List, Literal, Self
from sqrtpy import PrettySqrt, PrettySqrtExpr, PrettyFraction

# TO-DO LIST:
#   + Complete the quadratic equation logic for the Equation.solve() method (DONE)
#   + Simplify the equation and the result of the equation when it is displayed
#   + Add docstrings to each method to explain how it works.


class EquationError(Exception):
    """Raise when an equation is shown to be invalid \n
    (e.g. coeff `a` is zero in a quadratic equation, weird symbols, etc)."""

    pass

class InvalidAssignmentError(Exception):
    ...
    
class Equation:
    """
    A class that solves simple symbolic equations.\n
    Currently only supports linear and quadratic equations.
    """
    number = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)"
    symbol = r"\*?[a-zA-Z]"
    exp_pattern = r"(?:\*\*\d+)"
    def __init__(self, equation: str):
        
        self.equation = equation.replace(" ", "").replace("**1", "")
        nested_operators = ["+-", "-+", "++", "--"]
        while any(x in self.equation for x in nested_operators):
            self.equation = (
                self.equation.replace("+-", "-")
                .replace("-+", "-")
                .replace("--", "+")
                .replace("++", "+")
            )
        simplify = self._quick_simplify()
        if hasattr(simplify, "equation"):
            self.equation = simplify.equation
        self.equation = self._remove_zeroes()
        self._symbol = self.symbol
        #self._parse = self.parse
    @property
    def symbol(self) -> str:
        """
        Return the character of the symbol in the equation.
        """
        def __separate(n: list) -> str:
            return "','".join(n) + "'"
        pattern = r"[a-zA-Z]"
        chars = findall(pattern, self.equation)
        chars_set = set(chars)
        if (i := len(chars_set)) > 1 or i < 1:
            raise EquationError(
                f"Equation must have only one symbol, found {i}:"
                    f"{__separate(sorted(list(chars_set)))}"
            )
        elif special := findall(r"[^a-zA-Z0-9+\-*=.]", self.equation):
            raise EquationError(
                "Invalid special character(s) found in equation: '"
                + __separate(special)
            )
        else:
            return chars[0]

    @staticmethod
    def _get_degree(expr):
        # Find any matches that immediately precedes the equation symbol
        # with the double asterisks
        pattern = r"(?<=[a-zA-Z]\*\*)\d+"
        exp = findall(pattern, expr)
        deg = max(int(x) for x in exp) if exp else 1
        if deg == 0:
            raise ZeroDivisionError("Equation cannot have zero as its largest degree")
        return deg

    @property
    def degree(self) -> int:
        """Return the largest degree of the equation."""

        return Equation._get_degree(self.equation)

    def __repr__(self):
        return self.__str__().replace(" = 0", "", 1)

    def __str__(self):
        exprs = self.find()
        if exprs[0].startswith("+"):
            exprs[0] = exprs[0].replace("+", "", 1)
        return f"{' '.join(exprs)} = 0"

    def __find(self, b_pattern=False, get_c=False, get_d=False):
    
        if b_pattern:
            return rf"{self.number}(?={self.symbol})"
        sym_pattern = rf"(?:{self.number}|[+-])?{self.symbol}{self.exp_pattern}?"
        sym = sorted(
            findall(sym_pattern, self.equation), reverse=True, key=Equation._get_degree
        )
        if get_c:
            remainder = self.equation
            for term in sym:

                remainder = remainder.replace(term, "", 1)
            return remainder
        # print("from __find():", sym)
        return sym

    def find(self) -> List[str]:
        """Return a list containing all expressions of an equation,
        in sorted order. All elements of this list are strings."""
        if self.__find(get_c=True):  # Check if c is provided
            sorted_order = self.__find() + [self.__find(get_c=True)]
        else:
            sorted_order = self.__find()

        ops = "+-"
        for x, y in enumerate(sorted_order):
            if x == 0:
                continue
            if all(c != y[0] for c in ops):
                sorted_order[x] = "+" + y
        return sorted_order
        
    
    
    @staticmethod
    def is_valid(eq: str, check_num = False) -> bool:
        """ 
        Check if an equation is valid or not. An equation is considered to be valid
        if it satisfies the following: \n 
        — The equation should only have only one symbolic term. \n
        — The equation must not have any unsupported special characters. \n
        — Operators, especially '*' and '=', should be placed correctly in the equation. \n\n
        
        
        If the string is not qualified as a valid equation and `check_num` is True, 
        This method will check if the given string is a valid number.
        """
        def isnumber(n):
            return (n.replace(".", "", 1)
            .replace("-", "", 1)).isdigit()
        
        if not check_num:
            try:
                Equation(eq)
            except EquationError:
                return False
            return True
        return isnumber(eq)
        
    def _remove_zeroes(self):
        expr = self.equation
    
        # 1) Remove terms like 0x, 000x, 0*x, x*0, including exponentiation: 0x**n
        expr = sub(
            r'(^|[+\-])(?:0+(?:\*?[a-zA-Z](?:\*\*\d+)?)|(?:[a-zA-Z](?:\*\*\d+)?\*0+))',
            r'\1',
            expr
        )
    
        # 2) Remove standalone zeros: +0, -0, +000, -000
        expr = sub(r'([+\-])0+(?=[+\-]|$)', r'\1', expr)
    
        # 3) Remove leading zeros in numeric coefficients (0005x → 5x)
        expr = sub(r'(^|[+\-])0+(?=\d)', r'\1', expr)
    
        # 4) Cleanup redundant operators
        expr = expr.lstrip("+")
        expr = sub(r'\+\-', '-', expr)
        expr = sub(r'\-\+', '-', expr)
        expr = sub(r'\+\+', '+', expr)
        expr = sub(r'\-\-', '+', expr)
    
        # 5) Fallback if everything vanished → 0
        if not expr or expr in "+-":
            expr = "0"
    
        self.equation = expr.rstrip("+").rstrip("-")
        return self.equation
        ...
    @staticmethod
    def from_parse(
        nums: List[int | float], 
        sym = "x"
        ) -> Self:
        """
        Given a list containing all coefficients of an equation, return
        an Equation that correctsponds to that parse.
        """
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
                #m = "*" if not remove_mul else ""
                exponent = f"{op}{c}{sym}{d}" 
                helper += exponent
            elif deg == 0:
                helper += str(nums[ind])
        
        return Equation(helper) 
    

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
                if Equation.is_valid(expr[0], check_num = True):
                    right = Equation(expr[1]).parse
                    left = [0]*(len(right) - 1) + [convert(expr[0])]
                elif Equation.is_valid(expr[1], check_num = True):
                    left = Equation(expr[0]).parse
                    right = [0] * (len(left) - 1) + [convert(expr[1])]
                else:
                    raise InvalidAssignmentError(f"Invalid equation: {self.equation}")
                    
            except ValueError as e:
                raise InvalidAssignmentError(
                    f"Expected 1 assignment operator '=', found {len(expr)-1}\n"
                    f"(Original error message: {e})"
                    )
            L, R = Equation.from_parse(left), Equation.from_parse(right)
            left_deg, right_deg = L.degree, R.degree
            if left_deg < right_deg:
                helper(left, right_deg - left_deg)
            else:
                helper(right, left_deg - right_deg)
            for x in range(len(left)):
                res.append(left[x] - right[x])
            return Equation.from_parse(res, sym = self.symbol)
        
            
    @property
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

        sym: list = self.__find()
        
        if len(sym) < self.degree:
            for x, y in enumerate(sym):
                if Equation._get_degree(y) != self.degree - x:
                    sym.insert(x, 0)
            sym.extend([0] * (self.degree - len(sym)))
        num = f_value(self.__find(get_c=True) or "0")
        for i in range(len(sym)):
            if isinstance(sym[i], str):
                sym[i] = get_value(sym[i], match(self.__find(b_pattern=True), sym[i]))

        return sym + [num]


    def solve(
        self, allow_imag=True
    ) -> PrettyFraction | dict[str, PrettySqrtExpr | int | PrettyFraction]:
        """Return a dictionary containing all real roots and imaginary roots
        of an equation (if possible). \n
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
        parsed_eq = self.parse
        try:
            if (divisor := gcd(*parsed_eq)) > 1:
                # Attempt to simplify the coefficients by dividing each coefficient
                # by the greatest common divisor of a, b, c (if it exists)
                for x in range(len(parsed_eq)):
                    parsed_eq[x] //= divisor
        except TypeError:
            pass

        if self.degree == 1:
            a, b = parsed_eq
            return PrettyFraction(-b, a)
        elif self.degree == 2:
            a, b, c = parsed_eq
            delta = b**2 - 4 * a * c
            denominator = 2 * a
            solution = {f"{self.symbol}1": None, f"{self.symbol}2": None}

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
                    f"Equation {self} has no real solutions \
                                    (allow_imag = False)"
                )
            if delta == 0:
                return {self.symbol: _safe_div(-b, 2 * a)}

            else:
                delta_root = PrettySqrt(delta)
                n = [
                    _safe_div(-b + delta_root, denominator),
                    _safe_div(-b - delta_root, denominator),
                ]
                for x, y in enumerate(solution.keys()):
                    solution[y] = n[x]
            return solution

        elif self.degree == 3:
            a, b, c, d = parsed_eq

            if a == 0:
                raise EquationError("Leading coefficient a cannot be zero for cubic")

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
                solution[f"{self.symbol}{i}"] = _simplify(r)
            return solution


def solve(
    equation: str, allow_imag=True
) -> PrettyFraction | dict[str, PrettySqrtExpr | int | PrettyFraction]:
    """
    This function is equivalent to the following method:
    ```python
    Equation(equation).solve(allow_imag: bool = True|False)
    ```. \n
    Return a dictionary containing all real roots and imaginary roots
    of an equation (if possible). \n
    All roots are simplified and are instances of the PrettySqrtExpr
    class (If the result contains a square root), which can be evaluated
    or used for calculating with other expressions. The single asterisk
    character ('*') can be used to imply multiplications between the
    coefficients and the symbol (optional).

    Example:
    ```python
        solve("x**2-12x+3")
        {'x1': 6 + sqrt(33), 'x2': 6 - sqrt(33)}
        solve("x**2 - 5x + 7")
        {'x1': 5/2 - 1/2*i*sqrt(3), 'x2': 5/2 + 1/2*i*sqrt(3)}
        solve("12x-7+22x**2") #Scrambled equation
        {'x1': -3/11 + 1/22*sqrt(190), 'x2': -3/11 - 1/22*sqrt(190)}
        solve("5y**2-15y+12") #Different symbol
        {'y1': 3/2 - 1/10*i*sqrt(15), 'y2': 3/2 + 1/10*i*sqrt(15)}
        ```
    """
    return Equation(equation).solve(allow_imag)


if __name__ == "__main__":
    #print(Equation("3x**3+5=2x**2+14x+2"))
    #print(Equation("3x+5=-2"))
    #print(Equation("3x+5-5+x**2-2x**2"))
    print(Equation("2x**2+2x+1 = 3x**2+3x+2"))
    print(Equation.from_parse([0, 1, 0, -1]))
    