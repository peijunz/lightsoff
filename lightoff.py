import numpy as np
import itertools as it

def GCD(a, b):
    m = np.array([1, 0])
    n = np.array([0, 1])
    while b:
        p, r = divmod(a, b)
        m, n = n, m-p*n
        a, b = b, r
    return a, m

def divmodr(b, m, n=None):
    if n is None:
        return divmod(b, m)
    k, (c, d) = GCD(m, n)
    p, r = divmod(b, k)
    return (p*c) % n, r

class RingBase:
    _base = 1
    _verbose = True

    def __init__(self, arg=0):
        if isinstance(arg, RingBase):
            self.val = arg.val % self._base
        else:
            self.val = arg % self._base

    def __int__(self):
        return int(self.val)

    def __str__(self):
        if self._verbose:
            return "{} (mod {})".format(self.val, self._base)
        else:
            return str(self.val)

    def __repr__(self):
        return str(self)

    def __add__(self, rhs):
        rhs = self.__class__(rhs)
        return self.__class__(self.val+rhs.val)

    def __radd__(self, lhs):
        lhs = self.__class__(lhs)
        return self.__class__(self.val+lhs.val)

    def __sub__(self, rhs):
        rhs = self.__class__(rhs)
        return self.__class__(self.val-rhs.val)

    def __rsub__(self, lhs):
        lhs = self.__class__(lhs)
        return self.__class__(lhs.val-self.val)

    def __mul__(self, rhs):
        rhs = self.__class__(rhs)
        return self.__class__(self.val*rhs.val)

    def __rmul__(self, lhs):
        lhs = self.__class__(lhs)
        return self.__class__(self.val*lhs.val)

    def __truediv__(self, rhs):
        rhs = self.__class__(rhs)
        p, r = divmodr(self.val, rhs.val, self._base)
        if r:
            raise ZeroDivisionError("Unsolvable congrunce division")
        return self.__class__(p)

    def __rtruediv__(self, lhs):
        lhs = self.__class__(lhs)
        return lhs/self

    def __floordiv__(self, rhs):
        rhs = self.__class__(rhs)
        p, r = divmodr(self.val, rhs.val, self._base)
        return self.__class__(p)

    def __rfloordiv__(self, lhs):
        lhs = self.__class__(lhs)
        p, r = divmodr(lhs.val, self.val, self._base)
        return self.__class__(p)

    def __mod__(self, rhs):
        rhs = self.__class__(rhs)
        p, r = divmodr(self.val, rhs.val, self._base)
        return self.__class__(r)

    def __rmod__(self, lhs):
        lhs = self.__class__(lhs)
        p, r = divmodr(lhs.val, self.val, self._base)
        return self.__class__(r)

    def __eq__(self, rhs):
        rhs = self.__class__(rhs)
        return rhs.val == self.val


def Ring(base, verbose=True):
    class _R(RingBase):
        _base = base
        _verbose = verbose
    return _R


def linear_solver(m, im, full=False):
    n = m.shape[0]
    im = im.reshape(n, -1)
    M = np.empty([n, n+im.shape[1]], dtype=m.dtype)
    M[:, :n] = m
    M[:, n:] = im
    nul = set()
    for x in range(n):
        for i in it.chain(range(x, n), nul):
            if M[i, x] != 0:
                break
        if M[i, x] == 0:
            nul.add(x)
            continue
        if x != i:
            M[[i, x]] = M[[x, i]]

        p = 1/M[x, x]
        if p is None:
            M[x, x:] /= M[x, x]
        else:
            M[x, x:] *= p

        for i in it.chain(range(x+1, n), nul):
            M[i, x:] -= M[x, x:]*M[i, x]

    s = sorted(set(range(n)) - nul, reverse=True)
    nul = sorted(nul)
    for k, x in enumerate(s):
        for i in s[k+1:]:
            if full:
                M[i, x:] -= M[x, x:]*M[i, x]
            else:
                M[i, n:] -= M[x, n:]*M[i, x]
    if full:
        return nul, M[:, n:], M[:, :n]
    else:
        return nul, M[:, n:]



def invr(m):
    '''Invert matrix with null space l'''
    im = np.vectorize(m[0, 0].__class__)(np.eye(m.shape[0], dtype='int'))
    return linear_solver(m, im)


R2 = Ring(2, verbose=False)
A2 = np.vectorize(R2)


def gmatrix(m, n):
    mat = np.zeros([m*n, m*n], dtype='int')
    for i in range(m):
        for j in range(n):
            ri = np.ravel_multi_index((i, j), (m, n))
            mat[ri, ri] = 1
            for ind in np.array(((i-1, j), (i+1, j), (i, j-1), (i, j+1))):
                if all(((0, 0) <= ind) & (ind < (m, n))):
                    ri2 = np.ravel_multi_index(ind, (m, n))
                    mat[ri, ri2] = 1
    return A2(mat)


def str2mat(s):
    rows = s.strip().split('\n')
    return np.array([list(r.strip()) for r in rows], dtype='int')


def lightsoff_solver(m, n):
    g = gmatrix(m, n)
    return invr(g)


def apply_sol(sol, s):
    if isinstance(s, str):
        s = str2mat(s)
    s = A2(s.flatten())
    nul, M = sol
    criteria = np.einsum('ij, j->i', M[nul], s)
    if not all(criteria == 0):
        return None
    return np.einsum('ij, j->i', M, s).reshape(5, 5)

def test_elimination():
    from fractions import Fraction
    a = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 3, 3], [0, 0, 2, 2]], dtype='double')
    a = np.vectorize(Fraction)(a)
    n, m = invr(a)
    print("Null space", n)
    print(m)

def test_first_row():
    solver = lightsoff_solver(5, 5)
    lights = np.zeros([25], dtype='int')
    row1 = ((0, 1),)*5
    num = 0
    for elem in it.product(*row1):
        lights[:5] = elem
        sol = apply_sol(solver, lights)
        if sol is not None:
            num += 1
            print('-'*30)
            print(lights.reshape(5, 5))
            print(sol)
    assert num == 8, "Wrong number of solutions"
    return solver

def interactive_solver():
    import argparse

    parser = argparse.ArgumentParser(description='Turn off all the lights')
    parser.add_argument('s', metavar='Puzzle', type=str, nargs=1,
                    help='a puzzle to solve for the accumulator')
    args = parser.parse_args()
    m = str2mat(args.s[0])
    solver = lightsoff_solver(*m.shape)
    print(apply_sol(solver, m))

if __name__ == "__main__":
    interactive_solver()
