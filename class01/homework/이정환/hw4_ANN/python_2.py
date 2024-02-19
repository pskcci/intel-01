#리스트
a = [1, 3, 4]
print(a)
a[0] = 9
print(a)

b = [1, 3, 'string']
print(b)
b.append(6.24)
print(b)

print(a*2)
print(b*2)
c = [a[i] + a[i] for i in range(len(a))]
print(c)

#튜플
a = (1, 2, 3)
print(a)
b = (1, 3, 'string')
print(b)

a[0] = 2
a.append(4)

#딕셔너리
info = {'A' : 2.3, 'B' : 'C', 5 : 'D'}
print(info)

info['A'] = 5.2
print(info)

info["Hello"] = [1, 2, 3, 4, 'World.']
print(info)
