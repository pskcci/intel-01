#String format 실습
print('{0} and {1}'.format('spam', 'eggs'))
print('{1} and {0}'.format('spam', 'eggs'))
print('This {food} is {adjective}.'.format(food='spam', adjective='absolutely horrible'))
print('The story of {0}, {1}, and {other}.'.format('Bill', 'Manfred', other='Georg'))

s = 'coffee'
n = 5
result = f'저는 {s}를 좋아합니다. 하루 {n}잔 마셔요.'
print(result)
