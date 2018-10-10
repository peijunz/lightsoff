import numpy as np
import itertools as it
import ring
import linear_solver as lsol
import re


R2 = ring.Ring(2, verbose=False)
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
    return np.array([list(re.sub("\s", "", r)) for r in rows], dtype='int')


def lightoff_solver(m, n):
    return lsol.invr(gmatrix(m, n))


def light_sol(sol, s):
    if isinstance(s, str):
        s = str2mat(s)
    sh = s.shape
    s = A2(s.flatten())
    res, _ = lsol.apply_sol(sol, s)
    if res is not None:
        return res.reshape(sh)


def test_first_row():
    solver = lightoff_solver(5, 5)
    lights = np.zeros([5, 5], dtype='int')
    row1 = ((0, 1),)*5
    num = 0
    for elem in it.product(*row1):
        lights[0] = elem
        sol = light_sol(solver, lights)
        if sol is not None:
            num += 1
            print('-'*30)
            print(lights)
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
    solver = lightoff_solver(*m.shape)
    print(light_sol(solver, m))


def single_solve(s):
    m = str2mat(s)
    g = gmatrix(*m.shape)
    n, v, _ = lsol.linear_solver(g, A2(m.flatten()))
    if all(v[n] == 0):
        print(v.reshape(m.shape))
        return str(v.reshape(m.shape)).replace('[', ' ').replace(']', ' ')
    else:
        return "None"


if __name__ == "__main__":
    interactive_solver()
