import numpy as np
import itertools as it
import ring
import linear_solver as lsol
import re
from functools import lru_cache

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


def str2mat(s, delimiter):
    rows = s.strip().split(delimiter)
    return np.array([list(re.sub("\s", "", r)) for r in rows], dtype='int')

@lru_cache
def lightoff_solver(m, n):
    return lsol.invr(gmatrix(m, n))


def turnoff_lights(lights):
    if isinstance(lights, str):
        lights = str2mat(lights)
    elif isinstance(lights, list):
        lights = np.array(lights)        
    m, n = lights.shape
    lights = A2(lights.flatten())
    res, _ = lsol.apply_sol(lightoff_solver(m, n), lights)
    if res is not None:
        return res.reshape(m, n)


def test_first_row():
    lights = np.zeros([5, 5], dtype='int')
    row1 = ((0, 1),)*5
    num = 0
    for elem in it.product(*row1):
        lights[0] = elem
        sol = turnoff_lights(lights)
        if sol is not None:
            num += 1
            print(elem)
    assert num == 8, "Wrong number of solutions"


def interactive_solver():
    import argparse

    parser = argparse.ArgumentParser(description='Turn off all the lights! Solve puzzles in https://wiki.gnome.org/Apps/Lightsoff')
    parser.add_argument("puzzle", metavar='Puzzle', help='a puzzle to solve')
    parser.add_argument("-d", "--delimiter", help="set delimiter",
                    default='\n')
    args = parser.parse_args()
    m = str2mat(args.puzzle, args.delimiter)
    print(m)
    print(turnoff_lights(m))


if __name__ == "__main__":
    test_first_row()
    interactive_solver()
