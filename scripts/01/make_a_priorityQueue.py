# Use heapq to Implement a priority queue

import heapq

class Item:
    def __init__(self, name, priority):
        self.name = name
        self.priority = priority
    def __repr__(self):
        return 'Item({!r})'.format(self.name, self.priority)

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item):
        heapq.heappush(self._queue, (-item.priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

q = PriorityQueue()
q.push(Item('5', 5))
q.push(Item('3', 3))
q.push(Item('8', 8))
q.push(Item('4', 4))
q.push(Item('0', 0))

print(q.pop())
print(q.pop())
print(q.pop())
print(q.pop())
print(q.pop())

