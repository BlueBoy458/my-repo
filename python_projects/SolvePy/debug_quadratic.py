from quadratic_pro import Equation
from timeit import timeit

        
a = Equation("5x**3+5x**2+8")._Equation__find()

if __name__ == "__main__":
    def run(test_output=False, t = 1):
        def test(a, number):
            expr = Equation(a)
            print("-----------------------------")
            print(f"{number}. Unmodified equation: {a}")
            print(f"Expression: {expr}")
            try:
                print(f"Degree: {expr.degree}")
                print(f"Parse: {expr.parse}")
                print(f"Is valid?: {expr.is_valid()}")
                print(f"Solution: {expr.solve()}")
                print("-----------------------------")
            except Exception as e:
                print("An error occured!!!:", e)
        test_cases = [
            "2",
            "2x",
            "2x-1=3x",
            "x**3-3x+2=x**3-2x**2",
            "2x**2-17x+3=8x**3-22x+4",
            "3x-5=12x",
            "8x**2=2",
            "2x+2 = 2x-3",
            "12x=-3x**2+8",
            "18x**2+12x-6=0",
            "22x**3-12x",
            "2x**2=2x**2",
            "9x**3=9x**3+18x",
            "152x**2+8x-46=8",
            "x=2x**3-2+8x**2",
            "2x+8",
            "0x**3+5x**2-8x+3=0x**3-8x+7",
            "7x**2-5+3x=8x",
            "8x**3-7x+12+3x**2=0x",
            "2x+3=2x**2+3x+5",
            "17x**3=2",
            "8x**3=8x**3-1",
            "3z**2-12z+24",
            "x**3-88x**2+12",
            Equation("x**3-4.5x"),
            Equation("008x**2+ 5 + 8x + 0x**3"),
            Equation("09x**3+008x**2"),
            Equation.from_parse([5,8,7,6], symbolic="t"),
            Equation.from_parse([2, 3], [4, 5, 6]),
            Equation.from_parse(8, [5, 7, 2, 6]),
            Equation.from_parse([2, 5, 8], [2, 5, 6]),
            Equation.from_parse(0, 0),
            Equation.from_parse([2,-8,5]),
            Equation.from_parse([1, 2, 5], -3),
            Equation.from_parse(2, [3, 4, 5]),
            Equation.from_parse([2, 3], [2, 3, 5]),
            Equation.from_parse(2, [3, 4, 5]),
            Equation.from_parse([2,3,4,5], 1235, symbolic="n"),
            Equation.from_parse([8,9], [5,8,7,6], symbolic="t"),
            
            ]
        very_long_cases = [
            "888x**2+172x+567",
            "7777x**2+568x-32556",
            "1234x**3-555x**2+71823x-1",
            "88888x**3-000007866x**2-712345x-12345",
            "12345627x**2-8274635x=8x**3-812736",
            "0x**7+5x**3-2x**2=871263x**3-123456x+8123457x**2",
            "55556x**3-456x**2-1234=8765-1234567x+56432x**2",
            "8888888888x**3-67854321x**2+7865436274x",
            "12345675234x**3=871247385x**2-1234562738x",
            "1234567891x**3=1234567891x**3-876546x**2+123456x",
            Equation.from_parse([86472,98762], [55323,12348,2317,687], symbolic="t"),
            Equation.from_parse([821345,987612], [781243,819378,123754,6878123], symbolic="t"),
            Equation.from_parse([87654321, 12345678], [9876543, 78654321, 7654348, 123456789]),
            Equation.from_parse([876543219, 7654321, 8765434, 123456789]), 
            Equation.from_parse([87654321, 43215678], [981235463, 9827432321, 79824748, 123456789]),
            Equation.from_parse([123456, 1454568, 1234678])
            ] * 3
        def helper(x):
            n = Equation(x)
            _deg = n.degree
            _p = n.parse
            _valid = Equation.is_valid(n)
            #_absurd = n.is_absurd()
            _s = n.solve()
        def find_longest(cases):
            time_per_op = [0]
            current = ""
            current_max = 0
            for x in cases:
                def _():
                    helper(x)
                t = timeit(_, number=1)
                if t > max(time_per_op):
                    current = x
                    current_max = t
                    time_per_op.append(t)
            return (current, current_max)
            
        def _test():
            for x in test_cases:
                helper(x)
        def test_long_case():
            for x in very_long_cases:
                helper(x)
                
        def report(cases, t, repeat=10, op=5):
            time = round(timeit(cases, number = repeat), 2)
        
            equations = len(t) * repeat
            sec = round(time/repeat, 2)
            milisec = round(sec * 1000, 1)
            longest_op = find_longest(t)
            s = round(longest_op[1], 3)
            ms = s * 1000
            print((f"Total equations: {equations}"
            f"\nNumber of calculations: {equations * 5}"
            ))
            print(f"time: {time} seconds")
            
            print(f"Average time: {sec} seconds ({milisec} ms)")
            print(f"Avg time per {len(t)} operations: {round(sec/5, 2)} seconds ({round(milisec / 5, 2)} ms)")
            print(f"Avg time per operation: {round(sec / 5 / 34, 7)} seconds ({round(milisec / 5 / 34, 2)} ms)")
            print(f"Operation with the longest execution time: {longest_op[0]} with execution time {s} seconds ({ms} ms), total execution time approximately {round(s * repeat * 3, 2)} seconds ({round(ms * repeat * 3, 2)} ms)")
            
        if test_output:
            [test(x, i) for i, x in enumerate(test_cases)]
            [test(x, i) for i, x in enumerate(very_long_cases)]
        else:
            print("Testing normal test cases:")
            report(_test, test_cases, 10)
            print("\nTesting very long cases:")
            report(test_long_case, very_long_cases, 10)
    
    def test_input():
        while True:
            try:
                n = Equation(input("Equation (or type q to quit): "))
                if repr(n) == "q":
                    break
                print(
                f"Equation {n}\n"
                f"Is valid?: {n.is_valid()}\n"
                f"Degree: {n.degree}\n"
                f"Parse: {n.parse}\n"
                f"Solution: {n.solve()}"
                )
            except Exception as e:
                print("An exception occurred:", e)
                
    
    #run(True)
    #a = Equation("x**2-3.5")
    #print(a**2 + 8*a - 15)
    #a = Equation(input("Equation 1: "))
    #b = Equation(input("Equation 2: "))
    #print(a % b)
    n = "6x**5 - 2x**4 - 15x + 1"
    a = Equation(n)
    b = Equation("3x**2 + 2x")
    print("Division result:",a / b)
    print("Modulus result:", a % b)
    print(a**2)
    #print(Equation("8x**2-3x+6")**15)
    #print(Equation(5) + a)
    #print(Equation("x") - 2)
    #print(-Equation("2x-2"))