# range to array
print(list(range(9)));

l = []
l.append(1)
l.append(2)
l.append(1)
print(l)
l.sort()
print(l)

# loop increase
sum = 0
for i in range(10001):
    sum += i
print('Sum of 0~10000 is: ', sum)

n = 100
sum = 0
while n > 0:
    if n < 10 and n % 2 == 0:
       sum += n
    n -= 1
print(sum)


