import numpy as np

def GCD(a, b):
    m = np.array([1, 0])
    n = np.array([0, 1])
    while b:
        p, q = divmod(a, b)
        m, n = n, m-p*n
        a, b = b, q
    return a, m[0]

def divmodr(b, m, n=None):
    if n is None:
        return divmod(b, m)
    k, c = GCD(m, n)
    p, q = divmod(b, k)
    return (p*c) % n, q

def divr(b, m, n=None):
    p, q = divmodr(b, m, n)
    if q:
        raise ZeroDivisionError("Unsolvable congrunce division")
    return p


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
        p = divr(self.val, rhs.val, self._base)
        return self.__class__(p)

    def __eq__(self, rhs):
        rhs = self.__class__(rhs)
        return rhs.val == self.val


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
    for i in range(n):
        for j in range(i, n):
            if M[j, i] != 0:
                break
        if j != i:
            M[[i, j]] = M[[j, i]]
        if M[i, i] == 0:
            continue
        M[i] = M[i]/M[i, i]
        for k in range(j+1, n):
            M[k] = M[k] - M[i]*M[k, i]
    l = []
    for i in range(n-1, -1, -1):
        if M[i, i] == 0:
            l.append(i)
            continue
        for j in range(i-1, -1, -1):
            M[j] = M[j] - M[i]*M[j, i]
    return M[:, n:], l


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
    s = A2(str2mat(s).flatten())
    M, l = sol
    criteria = np.einsum('ij, j->i', M[l], s)
    if not all(criteria == 0):
        return None
    return np.einsum('ij, j->i', M, s).reshape(5, 5).astype('int')


if __name__ == "__main__":
    solver = lightsoff_solver(5, 5)
    q = '''
        11100
        00000
        00000
        00000
        00000
        '''
    print(apply_sol(solver, q))
