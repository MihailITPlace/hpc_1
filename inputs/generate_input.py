import sys
import numpy as np
from numpy import linalg as LA

matrix_size = int(sys.argv[1])
matrix = np.random.random((matrix_size, matrix_size))
matrix = (matrix + matrix.T) / 2

np.savetxt(f'matrix-{matrix_size}', matrix, fmt='%lf')
w, _ = LA.eig(matrix)
np.savetxt(f'matrix-{matrix_size}-answers', w, fmt='%lf')
