#for loop
items = [1, 2, 3, 4, 'Hello', 6.24]

for k in range(0, len(items)):
    print(items[k])
print("============")
for item in items:
    print(item)
print("============")
items = [[1,2], [3,4], [5,6]]
for item in items:
    print(item[0], item[1])
print("============")
for item1, item2 in items:
    print(item1, item2)
print("============")
info = {'A' : 1, 'B' : 2, 'C' : 3}
for key in info:
    print(key, info[key])
print("============")
for key, value in info.items():
    print(key, value)

#zip이 들어간 for loop
items1 = [[1,2], [3,4], [5,6]]
items2 = [['A','B'], ['C','D'], ['E','F']]
print(items1)
print(items2)
print("=====================================")
for digits, characters in zip(items1, items2):
    print(digits, characters)

#한 줄 for loop
a = []
for k in range(0, 5):
    a.append(k)
print(a)
print("=====================================")
a = [k for k in range(0, 5)]
print(a)
print("=====================================")
a = [k if (k+1)%2 else k*5+1 for k in range(0, 5)]
print(a)
print("=====================================")
a = [k for k in range(0, 5) if k%2 == 0]
print(a)
print("=====================================")
a = {k : k+10 for k in range(0, 5)}
print(a)
print("=====================================")
a = [1, 3, 4]
c = [a[i] + a[i] for i in range(len(a))]
print(c)
