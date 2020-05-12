import numpy as np 
from scipy import sparse
from scipy.stats import unitary_group
from scipy.sparse import csr_matrix, coo_matrix
from circuit_dynamics_init import kron_raw
import cppimport

# import c++ code module
cxx = cppimport.imp("eigen_dot")

# implement sparse-mat with dense vec dot product in parallel using openmp
def unitary_cxx_parallel(wave, i, l): 
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
        wave = cxx.dot(i, l, un, wave, 1) 
        wave = np.reshape(wave,(2, 2**(l-2), 2))
        # shift the position and flatten array
        wave = np.moveaxis(wave, -1, 0).ravel(order='C')
        # pss = temp

    return wave

# random unitary evolution
def unitary_conventional_cxx(wave, i, l):  
    '''
    the conventional protocol that generate a vast sparse matrix,
    which is a kronecker product between a random 4*4 unitary with a sparse identity matrix
    of size 2^{l-2} then applying this matrix to the wavefunction
    '''
    if i > l/2:
        raise ValueError("Invalid input, i must be less or equal to l/2." )
    d_a = np.ones(2**(2*i))
    r_a = np.arange(2**(2*i))
    c_a = r_a
    shape_a = 2**(2*i)
    u = coo_matrix(unitary_group.rvs(4))
    d_b = u.data
    r_b = u.row
    c_b = u.col
    shape_b = u.shape[0]
    temp = kron_raw(d_a, r_a, c_a, d_b, r_b, c_b, shape_b)
    
    if i < l//2-1: # 2nd kronecker product with identity matrix
        d_a = temp[0]
        r_a = temp[1]
        c_a = temp[2]
        shape_b = 2**(l-2*i-2)
        d_b = np.ones(2**(l-2*i-2))
        r_b = np.arange(2**(l-2*i-2))
        c_b = r_b
        temp = kron_raw(d_a, r_a, c_a, d_b, r_b, c_b, shape_b)
    
    # if i == l//2, there's no need to perform 2nd kronecker product with 1*1 identity matrix   
        
    un = csr_matrix((temp[0],(temp[1],temp[2])), shape=(2**l, 2**l))
    wave = cxx.dot_simple(un, wave)

    return wave

# unconventional scheme, replacing dot product with c++
# slowest method
def unitary_cxx(wave, i, l): 
    '''
    using pybind11 and c++ library Eigen to perform dot product (no openmp)
    '''
    if i >= l//2:
        raise ValueError("Invalid partition. i must be less than l/2." )
    temp = np.zeros((2**(2*i), 2**(l-2*i)),dtype='c16')
    pss = temp
    for j in range(l):
        wave_split = np.split(wave, 2**(2*i))
        u = unitary_group.rvs(4)
        un = sparse.kron(u, sparse.identity(2**(l-2*i-2)))
        for k in range(2**(2*i)):
            pss[k] = cxx.dot_simple(un, wave_split[k])

        wave = np.concatenate(pss) # gathering wavefunction
        wave = np.reshape(wave,(2, 2**(l-2), 2))
        # shift the position and flatten array
        wave = np.moveaxis(wave, -1, 0).ravel(order='C')
        pss = temp

    return wave