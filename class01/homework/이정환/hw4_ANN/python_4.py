#update tuple
thistuple = ("apple", "banana", "cherry")
print(thistuple)
y = ("orange",)
thistuple += y
print(thistuple)

thistuple = ("apple", "banana", "cherry")
print(thistuple)
y = list(thistuple)
y.append("orange")
thistuple = tuple(y)
print(thistuple)

#tuple unpack
fruits = ("apple", "banana", "cherry", "strawberry", "raspberry")
(green, yellow, *red) = fruits
print(green)
print(yellow)
print(red)

fruits = ("apple", "mango", "papaya", "pineapple", "cherry")
(green, *tropic, red) = fruits
print(green)
print(tropic)
print(red)
