#enumerate
x = ('apple', 'banana', 'cherry')
y = enumerate(x)
print(list(y))
print("=============================================")
for entry in enumerate(['A', 'B', 'C']):
    print(entry)
print("=============================================")
for i, letter in enumerate(['A', 'B', 'C']):
    print(i, letter)
print("=============================================")
for i, letter in enumerate(['A', 'B', 'C'], start=101):
    print(i, letter)

#파일 쓰기/읽기

#쓰기
filename = 'readme.txt'
file = open(filename, 'w')
file.write("Hello, World!")
file.close()

filename = 'readme.txt'
file = open(filename, 'r')
content = file.read()
print(content)
file.close()

filename = 'readme.txt'
with open(filename, 'r') as file:
    content = file.read()
    print(content)
