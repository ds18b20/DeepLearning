# Ref: https://stackoverflow.com/questions/5618878/how-to-convert-list-to-string
list1 = ['1', '2', '3']
str1 = ''.join(list1)

list1 = [1, 2, 3]
str1 = ''.join(str(e) for e in list1)