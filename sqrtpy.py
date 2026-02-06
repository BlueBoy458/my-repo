from math import sqrt
from fractions import Fraction
from typing import Dict, Union, Tuple


# -----------------------------
# Simplify sqrt(n)
# -----------------------------
def simplify_root(n: int) -> Tuple[int, int]:
    if n == 0:
        return 0, 0

    coeff = 1
    rad = abs(n)
    i = 2
    while i * i <= rad:
        while rad % (i * i) == 0:
            rad //= i * i
            coeff *= i
        i += 1

    return coeff, rad


class PrettyFraction(Fraction):
    def __repr__(self):
        return f"{self.__str__()}"


Number = Union[int, float, PrettyFraction]


# -----------------------------
# Symbolic radical expression
# -----------------------------
class PrettySqrtExpr:
    def __init__(self, terms: Dict[int, PrettyFraction] | None = None):
        self.terms: Dict[int, PrettyFraction] = terms or {}
        self._cleanup()

    # ---------- normalization ----------
    def _cleanup(self):
        cleaned = {}
        for rad, coeff in self.terms.items():
            if coeff == 0:
                continue
            cleaned[rad] = cleaned.get(rad, 0) + coeff
        self.terms = cleaned

        # If the expression reduced to a single rational term (rad==1),
        # keep it as a Fraction for internal consistency.
        if self.terms.keys() == {1}:
            # ensure the coefficient is a Fraction
            c = self.terms[1]
            if not isinstance(c, PrettyFraction):
                self.terms[1] = PrettyFraction(c)

    # ---------- constructors ----------
    @staticmethod
    def from_number(n: Number):
        return PrettySqrtExpr({1: PrettyFraction(n)})

    @staticmethod
    def from_sqrt(n: int, coeff: Number = 1):
        c, r = simplify_root(n)
        sign = -1 if n < 0 else 1
        rad = -r if n < 0 else r
        return PrettySqrtExpr({rad: PrettyFraction(coeff) * c * sign})

    # ---------- ordering ----------
    def _sort_key(self):
        def key(item):
            rad, coeff = item
            imag = rad < 0
            return (imag, abs(rad), coeff)

        return tuple(sorted(self.terms.items(), key=key))

    def __eq__(self, other):
        return (
            isinstance(other, PrettySqrtExpr) and self._sort_key() == other._sort_key()
        )

    def __lt__(self, other):
        if not isinstance(other, PrettySqrtExpr):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    # ---------- display ----------
    def __repr__(self):
        return self._to_string()

    def _to_string(self):
        if not self.terms:
            return "0"

        parts = []
        for rad, coeff in sorted(
            self.terms.items(), key=lambda x: (x[0] < 0, abs(x[0]))
        ):
            sign = "-" if coeff < 0 else "+"
            c = abs(coeff)

            if rad == 1:
                part = f"{c}"
            elif rad < 0:
                r = abs(rad)
                part = f"{c}*i*sqrt({r})" if c != 1 else f"i*sqrt({r})"
            else:
                part = f"{c}*sqrt({rad})" if c != 1 else f"sqrt({rad})"

            parts.append((sign, part))

        first_sign, first_part = parts[0]
        result = first_part if first_sign == "+" else "-" + first_part
        for s, p in parts[1:]:
            result += f" {s} {p}"
        return result

    # ---------- LaTeX ----------
    def to_latex(self):
        if not self.terms:
            return "0"

        parts = []
        for rad, coeff in sorted(
            self.terms.items(), key=lambda x: (x[0] < 0, abs(x[0]))
        ):
            c = coeff
            sign = "-" if c < 0 else "+"
            c = abs(c)

            if rad == 1:
                body = f"{c}"
            elif rad < 0:
                body = (
                    rf"{c} i \sqrt{{{abs(rad)}}}"
                    if c != 1
                    else rf"i \sqrt{{{abs(rad)}}}"
                )
            else:
                body = rf"{c} \sqrt{{{rad}}}" if c != 1 else rf"\sqrt{{{rad}}}"

            parts.append((sign, body))

        first_sign, first_body = parts[0]
        latex = first_body if first_sign == "+" else "-" + first_body
        for s, b in parts[1:]:
            latex += f" {s} {b}"
        return latex

    def _repr_latex_(self):
        return f"${self.to_latex()}$"

    # If the expression is only a rational number (no radical part),
    # return it as an `int` (if whole) or `Fraction`.
    def as_number(self):
        if not self.terms:
            return 0
        if set(self.terms.keys()) == {1}:
            val = self.terms[1]
            if not isinstance(val, PrettyFraction):
                val = PrettyFraction(val)
            if val.denominator == 1:
                return int(val.numerator)
            return val
        return None

    # ---------- numeric ----------
    def __float__(self):
        total = 0.0
        for rad, coeff in self.terms.items():
            if rad < 0:
                raise ValueError("Cannot convert imaginary value to float")
            total += float(coeff) * sqrt(rad)
        return total

    def __int__(self):
        return int(float(self))

    # ---------- arithmetic ----------
    def __add__(self, other):
        if isinstance(other, PrettySqrtExpr):
            terms = self.terms.copy()
            for r, c in other.terms.items():
                terms[r] = terms.get(r, 0) + c
            res = PrettySqrtExpr(terms)
            num = res.as_number()
            return num if num is not None else res
        if isinstance(other, (int, float, PrettyFraction)):
            return self + PrettySqrtExpr.from_number(other)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return PrettySqrtExpr.from_number(other) - self

    def __mul__(self, other):
        if isinstance(other, (int, float, PrettyFraction)):
            res = PrettySqrtExpr(
                {r: c * PrettyFraction(other) for r, c in self.terms.items()}
            )
            num = res.as_number()
            return num if num is not None else res

        if isinstance(other, PrettySqrtExpr):
            result = {}
            for r1, c1 in self.terms.items():
                for r2, c2 in other.terms.items():
                    sign = -1 if (r1 < 0) ^ (r2 < 0) else 1
                    coeff, rad = simplify_root(abs(r1 * r2))
                    rad = rad if sign > 0 else -rad
                    result[rad] = result.get(rad, 0) + c1 * c2 * coeff
            res = PrettySqrtExpr(result)
            num = res.as_number()
            return num if num is not None else res

        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float, PrettyFraction)):
            res = self * PrettyFraction(1, other)
            if isinstance(res, PrettySqrtExpr):
                num = res.as_number()
                return num if num is not None else res
            return res

        if isinstance(other, PrettySqrtExpr) and len(other.terms) == 1:
            ((r, c),) = other.terms.items()
            conj = PrettySqrtExpr({-r: c}) if r < 0 else PrettySqrtExpr({r: c})
            res = (self * conj) / (c * c * abs(r))
            if isinstance(res, PrettySqrtExpr):
                num = res.as_number()
                return num if num is not None else res
            return res

        return NotImplemented

    def __rtruediv__(self, other):
        return PrettySqrtExpr.from_number(other) / self

    def __pow__(self, power):
        if not isinstance(power, int):
            return NotImplemented

        # x**0 = 1
        if power == 0:
            return PrettySqrtExpr.from_number(1)

        # x**(-n) = 1 / (x**n)
        if power < 0:
            return PrettySqrtExpr.from_number(1) / (self ** (-power))

        # positive integer power
        result = PrettySqrtExpr.from_number(1)
        base = self
        exp = power

        # exponentiation by squaring (efficient)
        while exp > 0:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1

        return result

    def __rpow__(self, other):
        # number ** PrettySqrtExpr is not algebraic in general
        return NotImplemented


# -----------------------------
# Convenience constructor
# -----------------------------
def PrettySqrt(n: int):
    return PrettySqrtExpr.from_sqrt(n)


# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    a = PrettySqrt(8)
    b = PrettySqrt(12)

    print(a + a + PrettySqrt(12))
    print((a + b) * (a - b))
    print(b)
    print(a * b)
    print(1 / PrettySqrt(-2))
    print(PrettySqrt(280) - PrettySqrt(280))

    # print(2**PrettySqrt(28))
