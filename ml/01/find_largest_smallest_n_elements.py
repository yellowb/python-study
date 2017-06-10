# Use heapq to find the largest-n or smallest-n elements

import heapq

# Very simple example
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
print(heapq.nlargest(3, nums)) # Prints [42, 37, 23]
print(heapq.nsmallest(3, nums)) # Prints [-4, 1, 2]
print(nums) # will not modify original array

# Sort by customized key
portfolio = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}
]
cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])
print(cheap)
print(expensive)

# make array into a heap
heapq.heapify(nums)
print(nums)
heapq.heappop(nums)
print(nums)

# Get the max or min element from array, not use heapq
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
print(max(nums)) # Prints 42
print(min(nums)) # Prints -4
print(nums) # will not modify original array


