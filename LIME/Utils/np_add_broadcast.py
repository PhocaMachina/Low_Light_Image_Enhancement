import numpy
import numpy as np


def padded_add(a:np.ndarray, b:np.ndarray):
    m, n = a.shape
    p, q = b.shape
    x, y = max(m, p), max(n, q)
    # print(m,n,p,q,x,y)
    extended_a = numpy.zeros((x, y), complex)
    extended_b = numpy.zeros((x, y), complex)
    extended_a[0:m, 0:n] = a[0:m, 0:n]
    extended_b[0:p, 0:q] = b[0:p, 0:q]
    return extended_a + extended_b

if __name__ == '__main__':
    a = np.ones((9, 10))
    b = np.ones((5, 6))
    print(padded_add(a, b))