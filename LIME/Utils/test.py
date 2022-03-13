import numpy as np


def func(**kwargs):
    print(kwargs['mu0'])

def func1(**kwargs):
    def funcin(a:int = 1, b:int = 2):
        print(a,b)
    funcin(kwargs['a'], kwargs['b'])

if __name__ == '__main__':
    func(mu0=1)
    func1(b=32, a=12)