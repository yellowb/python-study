# Use deque as fixed length queue
# to keep last appended N elements.

from collections import deque

items = ['A', 'B', 'C', 'D', 'E']
queue = deque(maxlen=3)

for i in items:
    queue.append(i)
    print(queue)

print(queue)
