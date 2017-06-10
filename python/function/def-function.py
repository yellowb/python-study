# not good
def my_abs(x):
    if x >= 0:
        return x
    else:
        return -x

print(my_abs(-9))

# a better one with arg type checking
def my_better_abs(x):
    if not isinstance(x, (int, float)):
        raise TypeError('x should be int or float.')
    else:
        if x >= 0:
            return x
        else:
            return -x

# print(my_better_abs('0'))

# returns a tuple
def func_return_multiple(x):
    return (x, -x)

a1, a2 = func_return_multiple(9)
print(a1, a2)




