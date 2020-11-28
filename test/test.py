import linear_solver as LS
import numpy as np
A = np.array([[1,2,3], [4,5,6], [0, 0, 0]])
print(A)
A = LS.fractize(A)
ortho, space = LS.triangulate(A)
print(LS.solve_triangular(A, space=space, ortho=ortho))
print(A)
