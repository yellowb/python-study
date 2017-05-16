from pprint import pprint

# 用星号来表示变成元素个数
grades = [10, 20, 30, 40, 50, 60]
first, *middle, last = grades
print(middle)   # middle是一个array

# 星号用在循环中
records = [
    ('foo', 1, 2),
    ('bar', 'hello'),
    ('foo', 3, 4),
]

def do_foo(x, y):
    print('foo', x, y)

def do_bar(s):
    print('bar', s)

for tag, *args in records:
    if tag == 'foo':
        do_foo(*args)
    elif tag == 'bar':
        do_bar(*args)

# 星号用在字符串
head, *mid, tail = 'yellowb'
print(head)
print(mid)  # 一个Array, 元素是每一个字符: ['e', 'l', 'l', 'o', 'w']
print(tail)

# 忽略掉一些元素
record = ('ACME', 50, 123.45, (12, 18, 2012))
name, *_, (*_, year) = record   # _ 是一个变量, 不过后面不会再用到, 相当于把中间那些元素丢弃了
print(name, year)


