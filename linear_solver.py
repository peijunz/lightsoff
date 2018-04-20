'''Gaussian elimination method to solve linear equation or invert matrix'''
import numpy as np
import itertools as it


def linear_solver(A, b):
    '''Solve linear equations Ax=b with Gaussian elimination
    Args:
        A   Transformation matrix
        b   Inhomogenous term
    Return:
        ortho   The index that is not linearly independent
        solution
        kernel  The orthogonal complementary space of A

    Ref:
        https://en.wikipedia.org/wiki/Orthogonal_complement
        https://en.wikipedia.org/wiki/Kernel_(linear_algebra)
    '''
    n = A.shape[0]
    b = b.reshape(n, -1)
    type_ = A[0, 0].__class__
    M = np.empty([n, n+b.shape[1]], dtype=type_)
    M[:, :n] = A
    M[:, n:] = b
    ortho = []
    space = []
    for x in range(n):
        for i in it.chain(range(x, n), ortho):
            if M[i, x] != 0:
                break
        if M[i, x] == 0:
            M[x, x] = type_(-1)
            ortho.append(x)
            continue
        space.append(x)
        if x != i:
            M[[i, x]] = M[[x, i]]

        p = 1/M[x, x]
        if p is None:
            M[x, x:] /= M[x, x]
        else:
            M[x, x:] *= p

        for i in it.chain(range(x+1, n), ortho):
            M[i, x:] -= M[x, x:]*M[i, x]

    space.reverse()
    for k, x in enumerate(space):
        for i in space[k+1:]:
            M[i, x:] -= M[x, x:]*M[i, x]
    return ortho, M[:, n:], M[:, ortho]


def apply_sol(sol, b):
    '''Apply solution to inhomogenous vector term b
    Args:
        sol     Solution returned by linear_solver
        b       inhomogenous vector term
    Return:
        res     solution vector
        ortho   General solution
    '''
    ortho, M, ker = sol
    res = np.einsum('ij, j->i', M, b)
    if not all(res[ortho] == 0):
        return None
    return res, ker


def invr(m):
    '''Invert matrix with null space l'''
    im = np.vectorize(m[0, 0].__class__)(np.eye(m.shape[0], dtype='int'))
    return linear_solver(m, im)


def test_elimination():
    from fractions import Fraction
    a = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 3, 3],
                  [0, 0, 2, 2]], dtype='double')
    a = np.vectorize(Fraction)(a)
    ortho, sol, ker = invr(a)
    print("Null space", ortho)
    print("Solution", sol, sep='\n')
    print("Kernel space", ker, sep='\n')


if __name__ == "__main__":
    test_elimination()
