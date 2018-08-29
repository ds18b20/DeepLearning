#!/usr/bin/env python
# -*- coding: UTF-8 -*-


class Foo(object):
    def __init__(self):
        self.a = "xxx"
        self.b = "yyy"


f = Foo()
print(f.__class__.__name__)
print(f.__dict__)
print('{a}, {b}'.format(**f.__dict__))
