from mpi4py import MPI
import numpy as np
from scipy import sparse
from circuit_dynamics_init import ent, kron_raw, logneg, measure, unitary
from timeit import default_timer as timer

start = timer()
# time evolution consists of random unitaries + projective measurement
def evo(steps, wave, prob, L, n, partition):
    von = np.zeros(steps, dtype='float64') # von-Neumann entropy
    renyi = np.zeros(steps, dtype='float64') # Renyi entropy
    neg = np.zeros(steps, dtype='float64') # logarithmic negativity
    mut = np.zeros(steps, dtype='float64') # mutual information using von-Neumann entropy
    mutr = np.zeros(steps, dtype='float64') # mutual information in terms of Renyi entropy
    
    for t in range(steps):
        # evolve over odd links
        for i in range(L//2):
            wave = unitary(wave, i, L)     
        
        # measurement layer
        for i in range(L):
            wave = measure(wave, prob, i, L)

        # before evolve on even link, we need to rearrange indices first to accommodate the boundary condition PBC
        wave = np.reshape(wave,(2, 2**(L-2),2))
        # move the last site into the first one such that the unitaries can connect the 1st and the last site
        wave = np.moveaxis(wave,-1,0)
        wave = wave.flatten()
        
        # evolve over even links
        for i in range(L//2):
            wave = unitary(wave, i, L)  

        #shift the index back to the original order after evolution
        wave = np.reshape(wave,(2, 2, 2**(L-2)))
        wave = np.moveaxis(wave,-1,0)
        wave = np.moveaxis(wave,-1,0).flatten()

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

    return np.array([von, renyi, neg , mut, mutr])

#initializing the wavefunction for 2*L steps to achieve steady state 
def initial_evo(steps, wave, prob, L, n, partition):  
    for t in range(steps):
        # evolve over odd links
        for i in range(L//2):
            wave = unitary(wave, i, L)     
        
        # measurement layer
        for i in range(L):
            wave = measure(wave, prob, i, L)

        # before evolve on even link, we need to rearrange indices first to accommodate the boundary condition PBC
        wave = np.reshape(wave,(2, 2**(L-2),2))
        # move the last site into the first one such that the unitaries can connect the 1st and the last site
        wave = np.moveaxis(wave,-1,0)
        wave = wave.flatten()
        
        # evolve over even links
        for i in range(L//2):
            wave = unitary(wave, i, L)  

        #shift the index back to the original order after evolution
        wave = np.reshape(wave,(2, 2, 2**(L-2)))
        wave = np.moveaxis(wave,-1,0)
        wave = np.moveaxis(wave,-1,0).flatten()

        #measurement layer
        for i in range(L):
            wave = measure(wave, prob, i, L)
       
    return wave 


# reading parameters from file
para = open('para_haar.txt', 'r')
para = para.readlines()
# the paramters are system size, measurement probability and discrete time steps
L,  pro, time = int(para[0]), float(para[1]), int(para[2])

# system partition
# with PBC, we partition system into 4 parts where a and b separated by c1 and c2
# c1 and c2 are effectively connected, so the system is composed of A, B and C
lc1, la, lb = int(np.floor(L/8)), int(np.floor(L/4)), int(np.floor(L/4))
lc2 = L-lc1-la-lb
# pack the partition into array
partition = np.array([L, la, lb, lc1, lc2], dtype="int64")

# initializing wavefunctions
p1 = np.ones(1)
p2 = np.zeros(2**L-1,dtype='c16')
# a product state with all spins align up
psi = np.concatenate((p1,p2),axis=0).T


# get the "steady" wavefunction
psi = initial_evo(2*L, psi, pro, L, 2, partition)

# MPI session
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Broadcasting the wavefunction to all nodes
if rank == 0:
    data = psi
else:
    data = np.empty(2**L, dtype='c16')

comm.Bcast(data, root=0)

# running multiple simulations concurrently
result = evo(time, data, pro, L, 2, partition)
print("hello from node %d" %rank)
print(result)

# set data receiving buffer
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, 5, time], dtype='float64')
# Gathering resulting numpy arrays
comm.Gather(result, recvbuf, root=0)


if rank == 0:
    result = np.mean(recvbuf, axis = 0) # get the averaged data and save
    np.savez_compressed('evo_L=%s_p=%s_t=%s'%(L, pro, time+2*L), ent=result[0], renyi=result[1], neg=result[2], mut=result[3], mutr=result[4])
    end = timer()
    print("Elapsed = %s" % (end - start))