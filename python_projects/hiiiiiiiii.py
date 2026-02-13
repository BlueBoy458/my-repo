from re import search, compile

__number = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)"
#__zero = compile(__number.replace(r"\d", "0", 1))
__zero = compile(r"[+-]?0+\.\d*")
n = "0.00"
print(__zero.findall(n))