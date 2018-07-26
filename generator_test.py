# coding=utf-8

#使用()而不是[]可以生成generator
L = [x*x for x in range(10)]
L
L = (x*x for x in range(10))
L
for n in L:
    print n
print "abc"
#在函数中使用yield可以生成generator
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        # yield b
        a, b = b, a+b
        n = n+1

fib(9)

for n in fib(9):
    print n

print L
