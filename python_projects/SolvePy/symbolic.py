from quadratic_pro import Equation

class Symbolic(Equation):
    #_Equation__allow_symbols = True
    def __init__(self, equation):
        super().__init__(equation)
        Equation._unrestrict_symbols()
        #print(self.find())
        
        #super().__init__("x")
        ...
        #print(Symbolic.sym_pattern.findall(self.equation))
    @property
    def symbol(self):
       return Symbolic.pattern.findall(self.equation)
      
      
    def find_vars(self):
        def helper(x):
            try:
                return ord(Equation(x).symbol)
            except TypeError:
                return 130
        all_terms = Symbolic.sym_pattern.findall(self.equation)
        constant = self.equation
        for x in all_terms:
            constant = constant.replace(x, "")
            
        print(all_terms)
        #sorted_terms = sorted(all_terms, key=helper)
        
        return sorted(all_terms + [constant], key=helper)
        ...
        
    def __str__(self):
        exprs = [str(x) for x in self.find_vars()]
        n = ""
        for i, x in enumerate(exprs):
            if i == 0:
                n += x
                continue
            if not any(x.startswith(ops) for ops in "+-"):
                x = "+" + x
            n += x.replace("+", " + ").replace("-", " - ")
        return n
n = Symbolic("xyz**2 - 8y + 8 + 3 + 7x")
print(n)
print(n.find_vars())
#print(n.equation)