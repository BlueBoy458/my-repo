
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

a = [5, 8, 3, 7, 6, 8, 7]
b = [5, 8, 7, 7] 
print(helper(a, b))
    