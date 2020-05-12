import numpy as np 
from scipy import sparse
from scipy.sparse.linalg import eigsh, svds
from scipy.stats import unitary_group
import random
from numba import jit
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import svdvals,svd


@jit(nopython=True, parallel=True, nogil=True)
def kron_raw(d_a, r_a, c_a, d_b, r_b, c_b, shape_b):
    '''
    This function computes the kronecker product between two sparse matrices in COO format.
    The algorithm remains the same with Scipy's approach. 
    We made some small changes to remove some overhead and force all inputs data
    to be numpy arrays to take advantange of numba, with a speed up around 2X. 
    '''
    # nz = # of stored values, including explicit zeros

    # use COO format    
    nz = len(d_b)
 
    # expand entries of a into blocks
    row = np.repeat(r_a, nz)
    col = np.repeat(c_a, nz)
    data = np.repeat(d_a, nz)
 
    # multiply the original coordinate to get the expanded coordinate
    row *= shape_b
    col *= shape_b

    # increment block indices
    
    row, col = row.reshape(-1,nz), col.reshape(-1,nz)
    
    # this can be verified from the definitions of kronecker product, see Wiki for the formula
    row += r_b
    col += c_b
    row, col = row.reshape(-1), col.reshape(-1)
    # compute block entries
    data = data.reshape(-1,nz) * d_b
    data = data.reshape(-1)

    return data, row, col


# von-Neumann and Renyi entanglement entropy
def ent(wave, n, la, l):
    lb = l-la
    # convert the wavefunction into a matrix for SVD
    temp = np.reshape(wave,(2**la, 2**lb))
    # SVD for entanglement entropy, only singular values calculated
    sp = np.linalg.svd(temp, compute_uv=False)
    tol = 1e-10
    # chop small singular values to zero to avoid numerical instability
    sp[abs(sp) < tol] = 0.0
    # choose only non-zero values to avoid feeding to log function
    sp = sp[np.nonzero(sp)]
    el = sp**2
    von = -np.dot(el,np.log2(el))
    ren = (1/(1-n))*np.log2(np.sum(el**(n)))
    # chop small values to zero
    if (abs(von) < tol):
        von = 0
    if (abs(ren) < tol):
        ren = 0
    # EE in log2 base
    return von, ren


def ent_approx(wave, n, la, l): 
    # approximated von-Neumann and Renyi entanglement entropy 
    # by keeping l largest singluar values
    # this approximation applies only when calculating half-chain EE
    lb = l - la
    tol = 1e-8
    # convert the wavefunction into a matrix for SVD
    temp = np.reshape(wave,(int(2**la),int(2**lb)))
    if lb != l//2: 
        sp=np.linalg.svd(temp, compute_uv=False) # calculating all singular values
        
    else: 
        sp = svds(temp, k = l, which='LM', return_singular_vectors=False) # return largest l singular values
    
    # chop small singular values to zero for numerical stability
    sp[abs(sp) < tol] = 0.0
    sp = sp[sp != 0]        
    el = sp**2
    von = -np.dot(el,np.log2(el))
    ren = (1/(1-n))*np.log2(np.sum(el**(n)))
    if (abs(von) < tol):
        von = 0
    if (abs(ren) < tol):
        ren = 0
    # EE in log2 base
    return von, ren


# logarithmic negativity and mutual information
def logneg(wave, n, partition):

    l, la, lb, lc1, lc2 = partition[0], partition[1], partition[2], partition[3], partition[4] 

    # region A
    ps = np.reshape(wave, (2**lc1, 2**la, 2**lc2, 2**lb))
    ps = np.moveaxis(ps,0,1)
    ps = np.reshape(ps,(2**la, 2**(l-la)))
    # entanglement entropy in region A
    en = ent(ps, n, la, l)  
    # sa and sar stand for von-Neumann and Renyi entanglement entropies
    sa, sar = en[0], en[1]
    

    # region B
    ps = np.reshape(wave, (2**(l-lb), 2**lb))
    en = ent(ps, n, l-lb, l)
    sb, sbr = en[0], en[1]

    # region C
    # since C composed of c1 and c2, we need to re-arrange the index to combine c1 and c2 into
    # a connected region
    ps = np.reshape(wave, (2**lc1, 2**la, 2**lc2, 2**lb))
    ps = np.moveaxis(ps,1,2)
    ps = np.reshape(ps,(2**(lc1+lc2), 2**(la+lb)))
    en = ent(ps, n, lc1+lc2, l)
    sc, scr = en[0], en[1]
    
    # log(negativity)
    rab = np.dot(ps.T,np.conjugate(ps)) #reduced density matrix by tracing out C
    # reshape the reduced density matrix to have 4 indices to facilitate partial transpose
    rab = np.reshape(rab,(2**la, 2**lb, 2**la, 2**lb))

    # partial transpose on A
    pab = np.moveaxis(rab, 0, 2)
    # rearrange indices to make pab into a matrix
    pab = pab.reshape(2**(la+lb), 2**(la+lb))
    # SVD of partial transposed density matrix
    sp = np.linalg.svd(pab, compute_uv=False)
    # definition of logarithmic negativity
    logn = np.log2(np.sum(sp))
    tol = 1e-5
    # returns logarithmic negativity and two mutual information
    result = np.array([logn, sa+sb-sc, sar+sbr-scr])
    # chop small values to be zero
    result[abs(result) < tol] = 0.0
    
    return result

# WRONG!!!
# approximate logarithmic negativity: incorrect
def logneg_approx(wave, n, partition):

    l, la, lb, lc1, lc2 = partition[0], partition[1], partition[2], partition[3], partition[4] 

    ps = np.reshape(wave, (2**lc1, 2**la, 2**lc2, 2**lb))
    ps = np.moveaxis(ps,1,2)
    ps = np.reshape(ps,(2**(lc1+lc2), 2**(la+lb)))

    # log(negativity)
    rab = np.dot(ps.T,np.conjugate(ps)) #reduced density matrix by tracing out C
    # reshape the reduced density matrix to have 4 indices to facilitate partial transpose
    rab = np.reshape(rab,(2**la, 2**lb, 2**la, 2**lb))

    # partial transpose on A
    pab = np.moveaxis(rab, 0, 2)
    # rearrange indices to make pab into a matrix
    pab = pab.reshape(2**(la+lb), 2**(la+lb))
    
    # The singular values of matrix m can be obtained from Sqrt[Eigenvalues[ConjugateTranspose[m].m]]
    # approximation: keeping only l largest eigenvalues
    pab = sparse.csr_matrix(np.dot(pab.conj().T, pab))
    #print(pab.shape)
    sp = np.array(eigsh(pab, k = l, which='LM',return_eigenvectors=False))
    tol = 1e-10
    sp[abs(sp) < tol] = 0.0
    sp = np.sqrt(sp)
    # definition of logarithmic negativity
    logn_approx = np.log2(np.sum(sp))
    
    if abs(logn_approx) < tol:
        return 0.0
    else:
        return logn_approx


# projective single-site measurement in z-basis
def measure_slow(wave, prob, l):
    '''
    the z-projection operator for each location is constructed
    in a straightforward manner by making tensor product between s_z and 
    identity matrix
    '''
    # there are two possible outcomes
    choice = [0, 1]
    for n in range(l):
        op=np.random.choice(choice, 1, p=[1-prob, prob]) #determine if to measure on the site
        # if the measurement is chosen at given position
        if op[0]==1:
            # projection operator for spin up in z-basis
            up=sparse.kron(sparse.identity(2**n),sz)
            up=sparse.kron(up,sparse.identity(2**(l-n-1)))
            up=0.5*(up+sparse.identity(2**l)) 

            # projection operator for spin down
            down=sparse.kron(sparse.identity(2**n),sz)
            down=sparse.kron(down,sparse.identity(2**(l-n-1)))
            down=0.5*(sparse.identity(2**l)-down) 

            # projection of wavefunction into z-up state
            pup1=up.dot(wave)
            pup=(wave.conjugate().T).dot(pup1)
            pup=np.asscalar(pup.real)

            # projection of wavefunction into z-down state
            pdown1=down.dot(wave)
            # pdown=wave.conjugate().T.dot(pdown1)
            pdown=1-pup

            # probility of the measurement is determined by the projection 
            out = np.random.choice([0,1],1,p=[pup, 1-pup])

            # if the measurement projects the spin onto z-up state
            if out[0]==0:
                wave=(1/np.sqrt(pup))*pup1 # normalization of wavefunction
            else:
                wave=(1/np.sqrt(pdown))*pdown1

    return wave


# projective single-site measurement in z-basis
def measure(wave, prob, pos, l):
    '''
    the better solution for z-projection is by observing structure of the tensor product between
    two diagonal matrices. It's still diagonal and follows the pattern described below.
    '''
    # there are two possible measurement outcomes
    choice = [0, 1]
    op = np.random.choice(choice, 1, p=[1-prob, prob]) #determine if to measure on this site
    # if the measurement is chosen at the given position
    if op[0] == 1:
        # construct \sigma_z_i in the many-body basis
        temp = np.ones(2**(l-pos-1))
        pz = np.concatenate((temp, -temp))
        # repeat the pattern for 2**pos times        
        pz = np.tile(pz, 2**pos)
        '''
        we can construct the diagonal projection matrix as follows 
        pz = sparse.diags(np.tile(pz, 2**pos))
        but it's unnecessary since we can avoid the dot product between a diagonal matrix and 
        a vector by direct list multiplication
        
        alternatively, this can be done with the more conventional method
        pz = sparse.kron(sparse.identity(2**pos), sz, format="csr")
        pz = sparse.kron(pz,sparse.identity(2**(l-pos-1)), format="csr")
        but the kronecker product is slow in scipy
        '''
        
        '''
        the spin-up/dowm projection operator is given by $1/2(I \pm \sigma_z)$
        we compute \sigma_z dot \psi part only since the identity part is trivial.
        This is also for avoiding the plus operation between 
        two large sparse matrices which is slow numerically
        '''
        # projection of wavefunction
        temp = pz*wave
        '''
        instead we only need to compute the plus and minus between vectors
        '''
        pup1 = 0.5*(wave + temp)
        pdown1 = 0.5*(wave - temp)
        # expectation values
        #temp = (wave.conjugate().T).dot(temp).real
        temp = np.vdot(wave, temp).real
        #assert abs(temp-1) < 1e-5
        pup = 0.5 + 0.5*temp
        pdown = 1 - pup
        
        '''
        in case the wavefunction is close to product state, the measurement 
        might yield pup=1 or pdown=1. 
        To avoid possible numerical errors such as pup > 1 or pup < 0, 
        we manually set the probability 
        to be 1 or 0.
        '''
        if pup > 1 or pup < 0:
            try:
                if abs(pup-1) < 1e-4:
                    pup = 1.0
                    pdown = 0.0
                    wave = pup1
                elif abs(pup) < 1e-4:
                    pup = 0.0
                    pdown = 1.0
                    wave = pdown1
            except:
                raise Exception('unphysical measurement results!')
        '''
        probility of the measurement outcome is determined 
        by the expetation value of projection operator
        '''
        out = np.random.choice([0, 1], 1, p = [pup, pdown])

        # if the measurement projects the spin onto the z-up state
        if out[0] == 0:
            wave = (1/np.sqrt(pup))*pup1 # normalization of wavefunction
        else:
            wave = (1/np.sqrt(pdown))*pdown1
            
    return wave

# random unitary evolution
def unitary_conventional(wave, i, l):  
    '''
    the conventional protocol that generate a vast sparse matrix,
    which is a kronecker product between a random 4*4 unitary with a sparse identity matrix
    of size 2^{l-2} then applying this matrix to the wavefunction
    '''
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
        
    un = coo_matrix((temp[0],(temp[1],temp[2])), shape=(2**l, 2**l))
    wave = un.dot(wave.flatten())

    return wave

def unitary(wave, i, l): # different unitary evolution protocol
    '''
    this protocol makes use of the fact that the 2^l by 2^l sparse matrix 
    is in fact block diagonal with each block size being 2^(l-2*i). We can 
    fix position index i and split the wavefunction into chunks given by 
    2^l/(2^{l-2*i})= 2^{2*i} then performing the mat-vec product for each chunks,
    with the same sparse matix un. 

    Then we proceed to shift the position index of the wavefuntion and evolve it with 
    another random unitary. After shifting L times the position index returns its original 
    place and completes a whole evolution (both on even and odd links) 

    To reach optimal efficiency, the position index i must be chosen accordingly with the given
    system size l.
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
            pss[k] = un.dot(wave_split[k])

        wave = np.concatenate(pss) # gathering wavefunction
        wave = np.reshape(wave,(2, 2**(l-2), 2))
        # shift the position and flatten array
        wave = np.moveaxis(wave, -1, 0).ravel(order='F')
        pss = temp

    return wave


# generate a small data set to feed kron_raw() for compilation
d_a = np.ones(4)
r_a = np.arange(4)
c_a = r_a
shape_a = 4

d_b = d_a
r_b = r_a
c_b = c_a
shape_b = 4


if __name__ == "__main__":  # jit compile
    kron_raw(d_a, r_a, c_a, d_b, r_b, c_b, shape_b)  