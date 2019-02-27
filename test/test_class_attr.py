#!/usr/bin/env python
# -*- coding: UTF-8 -*-


# immutable
class Foo(object):
    def __init__(self):
        self.a = "xxx"
        self.b = "yyy"


# mutable
class AAA(object):
    def __init__(self, a):
        self.a = a
        self.y = BBB(self.a)

    def show(self):
        print(self.a)


class BBB(object):
    def __init__(self, b):
        self.b = b

    def show(self):
        print(self.b)


if __name__ == '__main__':
    print('### immutable ###')
    a0 = AAA(1.0)  # float object is immutable
    print('a0.a:\n', a0.a)
    print('a0.y.b:\n', a0.y.b)
    a0.a = 2.0
    print('a0.a:\n', a0.a)
    print('a0.y.b:\n', a0.y.b)

    print('### mutable ###')
    a = AAA([1.0, 2.0])  # list object is immutable
    print('a.a:\n', a.a)
    print('a.y.b:\n', a.y.b)
    a.a[0] = 5.0
    print('a.a:\n', a.a)
    print('a.y.b:\n', a.y.b)

    print('### id change ###')
    a.a = [3.0, 4.0]
    print('a.a:\n', a.a)
    print('a.y.b:\n', a.y.b)

    print('### init parameter ###')
    i = [8.0, 9.0]
    b = AAA(i)
    print('b.a:\n', b.a)
    i[0] = 7.0
    print('b.a:\n', b.a)