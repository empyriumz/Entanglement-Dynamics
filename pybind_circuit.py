import numpy as np 
from scipy import sparse
from scipy.stats import unitary_group
from scipy.sparse import csr_matrix
import cppimport


code = cppimport.imp("eigen_dot")

def unitary_cxx_parallel(wave, i, l): # different unitary evolution protocol
    '''
    using pybind11 and c++ library eigen to perform dot product 
    '''
    if i >= l//2:
        raise ValueError("Invalid partition. i must be less than l/2." )
    for j in range(l):
        u = unitary_group.rvs(4)
        # csr format for sparse matrix to enable built-in optimization of dot product in Eigen library
        un = sparse.kron(u, sparse.identity(2**(l-2*i-2)), format="csr")
        # call c++ functions to perform dot product in parallel
        wave = code.dot(int(i), int(l), un, wave) 
        wave = np.reshape(wave,(2, 2**(l-2), 2))
        # shift the position and flatten array
        wave = np.moveaxis(wave, -1, 0).ravel(order='F')
        # pss = temp

    return wave