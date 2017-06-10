# construct a set from a list
set1 = set([1, 1, 2, 3, 'abc', 'abc'])
print(set1)

# add key
set1.add(1)
print(set1)
set1.add(4)
print(set1)
set1.add((1, 2))
print(set1)

# remove key
set1.remove(1)
print(set1)
set1.remove((1, 2))
print(set1)

# 交集并集
set2 = {1, 2, 3}
set3 = {1, 2, 4}
print(set2 & set3)
print(set2 | set3)
print(set2 - set3)
print(set2 ^ set3)
