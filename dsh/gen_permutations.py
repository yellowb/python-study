from itertools import permutations

list = ['P1', 'P2', 'P3']
gen = permutations(list, len(list))

for line in gen:
    print(line)