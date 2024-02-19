#변수 선언
a = 1
b = 2.
c = "String"
print(a)
print(b)
print(c)
print(type(a))
print(type(b))
print(type(c))

#함수 선언
def f(x, y):
    val = x + y
    return val
a = 1
b = 2.
d = f(a, b)
print(d)

#익명 함수
f = lambda x,y : x + y
a = 1
b = 2.
d = f(a, b)
print(d)
