import numpy as np

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

    def __sub__(self, rhs):
        rhs = self.__class__(rhs)
        return self.__class__(self.val-rhs.val)

    def __mul__(self, rhs):
        rhs = self.__class__(rhs)
        return self.__class__(self.val*rhs.val)

    def __truediv__(self, rhs):
        rhs = self.__class__(rhs)
        p, r = divmodr(self.val, rhs.val, self._base)
        if r:
            raise ZeroDivisionError("Unsolvable congrunce division")
        return self.__class__(p)

    def __mod__(self, rhs):
        rhs = self.__class__(rhs)
        p, r = divmodr(self.val, rhs.val, self._base)
        return self.__class__(r)

    def __eq__(self, rhs):
        rhs = self.__class__(rhs)
        return rhs.val == self.val

    def inv(self):
        p, r = divmodr(1, self.val, self._base)
        if r == 0:
            return self.__class__(p)


def Ring(base, verbose=True):
    class _R(RingBase):
        _base = base
        _verbose = verbose
    return _R

@profile
def linear_solver(m, im):
    n = m.shape[0]
    im = im.reshape(n, -1)
    M = np.empty([n, n+im.shape[1]], dtype='object')
    M[:, :n] = m
    M[:, n:] = im
    col2row = np.zeros([n], dtype='int')
    row2col = np.zeros([n], dtype='int')
    x, y = 0, 0
    null = []
    while x < n and y < n:
        for i in range(x, n):
            if M[i, y] != 0:
                break
        if M[i, y] == 0:
            null.append(y)
            row2col[y] = x - y - 1
            y += 1
            continue
        if x != i:
            M[[i, x]] = M[[x, i]]
        p = M[x, y].inv()
        if p is None:
            M[x, y:] /= M[x, y]
        else:
            M[x, y:] *= p
        for i in range(x+1, n):
            M[i, y:] -= M[x, y:]*M[i, y]
        col2row[x] = y
        row2col[y] = x
        x += 1
        y += 1
    col2row = col2row[:x]
    #row2col = np.cumsum(row2col) - 1
    for i in range(x-1, -1, -1):
        y = col2row[i]
        for j in range(i-1, -1, -1):
            M[j, n:] = M[j, n:] - M[i, n:]*M[j, y]
            # Unnecessary operation
            # M[j, y:n] = M[j, y:n] - M[i, y:n]*M[j, y]
    M[list(range(n))] = M[row2col]
    return M[:, n:], null


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
    M, l = sol
    criteria = np.einsum('ij, j->i', M[l], s)
    if not all(criteria == 0):
        return None
    return np.einsum('ij, j->i', M, s).reshape(5, 5)

def test_elimination():
    a = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 3, 3], [0, 0, 2, 2]], dtype='int')
    m, n = invr(a)
    print(m, n)

def test_first_row():
    import itertools as it
    solver = lightsoff_solver(5, 5)
    lights = np.zeros([25], dtype='int')
    row1 = ((0, 1),)*5
    for elem in it.product(*row1):
        lights[:5] = elem
        sol = apply_sol(solver, lights)
        if sol is not None:
            print('-'*30)
            print(lights.reshape(5, 5))
            print(sol)
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
    test_first_row()
    #test_elimination()
    interactive_solver()
