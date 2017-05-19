from collections import defaultdict

# use array to store multiple values in same key
map1 = defaultdict(list)
map1['a'].append(1)
map1['a'].append(1)
map1['a'].append(2)
map1['b'].append(2)
print(map1['a'])

# use set to store multiple values in same key
map2 = defaultdict(set)
map2['a'].add(1)
map2['a'].add(1)
map2['a'].add(2)
map2['b'].add(2)
print(map2['a'])
