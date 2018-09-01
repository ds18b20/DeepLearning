#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import inspect
import os


def get_current_function_name():
    return inspect.stack()[1][3]


def get_attrs():
    print('Module:', __name__)
    print('File path: ', __file__)
    print('File name: ', os.path.basename(__file__))
    print('Current line No.: ', sys._getframe().f_lineno)
    print('Func name: ', sys._getframe().f_code.co_name)
    print('Func name: ', get_current_function_name())


class Foo(object):
    def __init__(self):
        self.a = "xxx"
        self.b = "yyy"


if __name__ == '__main__':
    # get function name
    get_attrs()
    print('****** line ******')
    # get class name & attributes
    f = Foo()
    print(f.__class__.__name__)
    print(f.__dict__)
    print('{a}, {b}'.format(**f.__dict__))
