map1 = {'name': 'Tom', 'age': 10}
print(map1.get('name'))     # if key not exists, return None
print(map1.get('weight'))

# add a key-value
map1['height'] = 100

# remove a key
map1.pop('name')
print(map1)

