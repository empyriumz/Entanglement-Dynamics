import cupy as cp
import numpy as np 
from scipy import sparse
from scipy.stats import unitary_group
import random
from timeit import default_timer as timer 
import cupyx as cpx
from cupyx.scipy.sparse import coo_matrix

# %% [code]
start = timer()
# von-Neumann and Renyi entanglement entropy
def ent(wave, n, L, la):
    lb = L-la
    # convert the wavefunction into a matrix for SVD
    temp = cp.reshape(wave,(2**la, 2**lb))
    # SVD for entanglement entropy, only singular values calculated
    sp = cp.linalg.svd(temp, compute_uv=False)
    tol = 1e-10
    # chop small singular values to zero to avoid numerical instability
    sp[abs(sp) < tol] = 0.0
    # choose only non-zero values to avoid feeding to log function
    sp = sp[cp.nonzero(sp)]
    el = sp**2
    von = -cp.dot(el,np.log2(el))
    ren = (1/(1-n))*cp.log2(np.sum(el**(n)))
    # chop small values to zero
    if (abs(von) < tol):
        von = 0
    if (abs(ren) < tol):
        ren = 0
    # EE in log2 base
    return von, ren

# logarithmic negativity and mutual information
def logneg(wave, n, partition):

    L, la, lb, lc1, lc2 = int(partition[0]), int(partition[1]), int(partition[2]), int(partition[3]), int(partition[4])

    # region A
    ps = cp.reshape(wave, (2**lc1, 2**la, 2**lc2, 2**lb))
    ps = cp.moveaxis(ps,0,1)
    ps = cp.reshape(ps,(2**la, 2**(L-la)))
    # entanglement entropy in region A
    en = ent(ps, n, L, la)  
    # sa and sar stand for von-Neumann and Renyi entanglement entropies
    sa, sar = en[0], en[1]
    

    # region B
    ps = cp.reshape(wave, (2**(L-lb), 2**lb))
    en = ent(ps, n, L, L-lb)
    sb, sbr = en[0], en[1]

    # region C
    # since C composed of c1 and c2, we need to re-arrange the index to combine c1 and c2 into
    # a connected region
    ps = cp.reshape(wave, (2**lc1, 2**la, 2**lc2, 2**lb))
    ps = cp.moveaxis(ps,1,2)
    ps = cp.reshape(ps,(2**(lc1+lc2), 2**(la+lb)))
    en = ent(ps, n, L, lc1+lc2)
    sc, scr = en[0], en[1]
    
    # log(negativity)
    rab = cp.dot(ps.T,cp.conj(ps)) #reduced density matrix by tracing out C
    # reshape the reduced density matrix to have 4 indices to facilitate partial transpose
    rab = cp.reshape(rab,(2**la, 2**lb, 2**la, 2**lb))

    # partial transpose on A
    pab = cp.moveaxis(rab, 0, 2)
    # rearrange indices to make pab into a matrix
    pab = pab.reshape(2**(la+lb), 2**(la+lb))
    # SVD of partial transposed density matrix
    sp = cp.linalg.svd(pab, compute_uv=False)
    # definition of logarithmic negativity
    logn = cp.log2(cp.sum(sp))
    tol = 1e-10
    # returns logarithmic negativity and two mutual information
    result = np.array([logn, sa+sb-sc, sar+sbr-scr])
    # chop small values to be zero
    result[abs(result) < tol] = 0.0
    
    return result

# %% [code]
def kron_raw(d_a, r_a, c_a, d_b, r_b, c_b, shape_b): # use COO format      
    # nz = # of stored values, including explicit zeros     
    nz = len(d_b)
 
    # expand entries of a into blocks
    row = cp.repeat(r_a, nz)
    col = cp.repeat(c_a, nz)
    data = cp.repeat(d_a, nz)
 
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

# projective single-site measurement in z-basis
def measure(wave, prob, pos, L):
    # there are two possible measurement outcomes
    choice = [0, 1]
    op = cp.random.choice(choice, 1, p=[1-prob, prob]) #determine if to measure on this site
    # if the measurement is chosen at the given position
    if op[0] == 1:
        # construct \sigma_z_i in the many-body basis
        temp = cp.ones(2**(L-pos-1))
        pz = cp.concatenate((temp,-temp))
        # repeat the pattern for 2**pos times        
        pz = cp.tile(pz, 2**pos)


        # projection of wavefunction

        temp = pz*wave
        pup1 = 0.5*(wave + temp)
        pdown1 = 0.5*(wave - temp)
        # expectation values
        #temp = (wave.conjugate().T).dot(temp)
        temp = cp.vdot(wave, temp)
        pup = 0.5 + 0.5*temp.real
        pdown = 1 - pup
        '''
        in case the wavefunction is close to product state, the measurement 
        might yield pup=1 or pdown=1. 
        To avoid possible numerical errors where pup>1, we manually set the probability 
        to be 1 or 0.
        '''
        if abs(pup-1)<1e-10: 
            pup = 1.0
            pdown = 0.0
            wave = pup1
        elif abs(pup)<1e-10:
            pup = 0.0
            pdown = 1.0
            wave = pdown1
        else:
            pdown = 1 - pup
            '''
            probility of the measurement outcome is determined 
            by the expetation value of projection operator
            '''
            out = np.random.choice([0,1],1,p=[pup, pdown])

            # if the measurement projects the spin onto the z-up state
            if out[0] == 0:
                wave = (1/cp.sqrt(pup))*pup1 # normalization of wavefunction
            else:
                wave = (1/cp.sqrt(pdown))*pdown1
            
    return wave

# %% [code]
# random unitary evolution
def unitary(wave, pos, L):    
    i = pos
    d_a = cp.ones(2**(2*i))
    r_a = cp.arange(2**(2*i))
    c_a = r_a
    shape_a = 2**(2*i)
    u = cp.array(unitary_group.rvs(4)) #cupy doesn't support generate random unitary yet
    d_b = u.flatten()
    r_b = cp.repeat(cp.arange(4), 4)
    c_b = cp.tile(cp.arange(4), 4)
    shape_b = u.shape[0]
    temp = kron_raw(d_a, r_a, c_a, d_b, r_b, c_b, shape_b)
    d_a = temp[0]
    r_a = temp[1]
    c_a = temp[2]
    shape_b = 2**(L-2*i-2)
    d_b = cp.ones(2**(L-2*i-2))
    r_b = cp.arange(2**(L-2*i-2))
    c_b = r_b
    t1 = kron_raw(d_a, r_a, c_a, d_b, r_b, c_b, shape_b)
    un = coo_matrix((t1[0],(t1[1],t1[2])), shape=(2**L, 2**L))
    wave = un.dot(wave.flatten())

    return wave

# %% [code]
# time evolution consists of random unitaries + projective measurement
def evo(steps, wave, prob, L, n, partition):
    von = cp.zeros(steps, dtype='float64') # von-Neumann entropy
    renyi = cp.zeros(steps, dtype='float64') # Renyi entropy
    neg = cp.zeros(steps, dtype='float64') # logarithmic negativity
    mut = cp.zeros(steps, dtype='float64') # mutual information using von-Neumann entropy
    mutr = cp.zeros(steps, dtype='float64') # mutual information in terms of Renyi entropy
    
    for t in range(steps):
        # evolve over odd links
        for i in range(L//2):
            wave = unitary(wave, i, L)     
        
        # measurement layer
        for i in range(L):
            wave = measure(wave, prob, i, L)

        # before evolve on even link, we need to rearrange indices first to accommodate the boundary condition PBC
        wave = cp.reshape(wave,(2, 2**(L-2),2))
        # move the last site into the first one such that the unitaries can connect the 1st and the last site
        wave = cp.moveaxis(wave,-1,0)
        wave = wave.flatten()
        
        # evolve over even links
        for i in range(L//2):
            wave = unitary(wave, i, L)  

        #shift the index back to the original order after evolution
        wave = cp.reshape(wave,(2, 2, 2**(L-2)))
        wave = cp.moveaxis(wave,-1,0)
        wave = cp.moveaxis(wave,-1,0).flatten()

        #measurement layer
        for i in range(L):
            wave = measure(wave, prob, i, L)
       
        result = ent(wave, n, L, L//2)
        von[t] = result[0]
        renyi[t] = result[1]
        result = logneg(wave, n, partition)
        neg[t] = result[0]
        mut[t] = result[1]
        mutr[t] = result[2]

    return von, renyi, neg , mut, mutr

# %% [code]
L,  pro, time = 18, 0.3, 20

# system partition
# with PBC, we partition system into 4 parts where a and b separated by c1 and c2
# c1 and c2 are effectively connected, so the system is composed of A, B and C
lc1, la, lb = int(np.floor(L/8)), int(np.floor(L/4)), int(np.floor(L/4))
lc2 = L-lc1-la-lb
# pack the partition into array
part = cp.array([L, la, lb, lc1, lc2], dtype="int64")

# initializing wavefunctions
p1 = cp.ones(1)
p2 = cp.zeros(2**L-1,dtype='c16')
# a product state with all spins align up
psi = cp.concatenate((p1,p2),axis=0).T


# %% [code]
#result = evo(time, psi, pro, L, 2, part)
print(result)
np.savetxt('haar_random_evo_L=%s_p=%s_t=%s'%(L, pro, time), result, delimiter=",")
end = timer()
print("Elapsed = %s" % (end - start))