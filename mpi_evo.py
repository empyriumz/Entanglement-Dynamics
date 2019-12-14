# from mpi4py import MPI
from timeit import default_timer as timer
from circuit_dynamics_init import *
import sys

# Global error handler
def global_except_hook(exctype, value, traceback):
    import sys
    try:
        import mpi4py.MPI
        sys.stderr.write("\n*****************************************************\n")
        sys.stderr.write("Uncaught exception was detected on rank {}. \n".format(
            mpi4py.MPI.COMM_WORLD.Get_rank()))
        from traceback import print_exception
        print_exception(exctype, value, traceback)
        sys.stderr.write("*****************************************************\n\n\n")
        sys.stderr.write("\n")
        sys.stderr.write("Calling MPI_Abort() to shut down MPI processes...\n")
        sys.stderr.flush()
    finally:
        try:
            import mpi4py.MPI
            mpi4py.MPI.COMM_WORLD.Abort(1)
        except Exception as e:
            sys.stderr.write("*****************************************************\n")
            sys.stderr.write("Sorry, we failed to stop MPI, this process will hang.\n")
            sys.stderr.write("*****************************************************\n")
            sys.stderr.flush()
            raise e

sys.excepthook = global_except_hook



start = timer()

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
part= np.array([L, la, lb, lc1, lc2], dtype="int64")

# initializing wavefunctions
p1 = np.ones(1)
p2 = np.zeros(2**L-1,dtype='c16')
# a product state with all spins align up
psi = np.concatenate((p1,p2),axis=0).T



def unitary_mpi(wave, i, l):
    # MPI session
    import mpi4py.MPI
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    shape_b = 2**(l-2*i-2)
    len_u = 16*shape_b # length of data array to be scattered to assemble unitary matrix
    for j in range(l):
        '''
        calculating the kronecker product between the random unitary matrix 
        and the identity matrix then broadcasting it to all nodes
        '''
        if rank == 0:
            #un = sparse.kron(u, sparse.identity(2**(l-2*i-2)), format='coo')
            u = coo_matrix(unitary_group.rvs(4), dtype = 'c16')
            d_a = u.data
            r_a = u.row
            c_a = u.col
            
            
            d_b = np.ones(shape_b)
            r_b = np.arange(shape_b)
            c_b = r_b
            un_coo = kron_raw(d_a, r_a, c_a, d_b, r_b, c_b, shape_b)
            #print(un_coo[0].shape, 2**(l-2*i), shape_b,len(r_a))
            
        # Broadcasting the unitary matrix to all nodes
            un_pack = np.array(un_coo, dtype='c16')
        else:
            un_pack = np.empty((3, len_u), dtype='c16')
        comm.Bcast(un_pack, root=0)
        assert un_pack.dtype == 'c16'
        # unpack the data and make the sparse matrix block
        un_sub = coo_matrix((un_pack[0],(un_pack[1],un_pack[2])), shape=(2**(l-2*i), 2**(l-2*i)))
        
        # Scatter wavefunction across the nodes
        sendbuf = None
        if rank == 0:
            sendbuf = np.array(np.split(wave, size), dtype='c16')
        # receiving buffer for the incoming chunked wavefunction
        recvbuf = np.empty(2**l//size, dtype='c16')
        # scatter chunked wavefunction from root node
        comm.Scatter(sendbuf, recvbuf, root=0)
        assert recvbuf.shape[0] == 2**l//size
        sub_blocks = 2**(2*i)//size # number of subblocks of the sparse matrix 
        # each sub-divided wavefunction are further splitted locally
        wave_split = np.split(recvbuf, sub_blocks)
        assert len(wave_split) == sub_blocks
        assert wave_split[0].dtype == 'c16'
        temp = np.empty_like(wave_split)
        assert temp.dtype == 'c16'
        # apply dot product for each block matrix
        for k in range(sub_blocks):
            temp[k] = un_sub.dot(wave_split[k])
        # stack wavefunction locally
        wave_split = np.concatenate(temp)
        assert wave_split.shape[0] == 2**l//size
        
        # set receiving buffer for root node
        recvbuf = None
        if rank == 0:
            recvbuf = np.empty([size, 2**l//size], dtype='c16')
        # gathering resulting wavefuntion
        comm.Gather(wave_split, recvbuf, root=0)
        if rank == 0:
            wave = recvbuf.ravel(order='F')
            assert wave.shape[0] == 2**l
            wave = np.reshape(recvbuf,(2, 2**(L-2), 2))
            # shift the axis to next position and flatten array
            wave = np.moveaxis(wave, -1, 0).ravel(order='F')

            return wave

def evo_parallel(steps, wave, prob, l, n, partition):
    von = np.zeros(steps, dtype='float64') # von-Neumann entropy
    renyi = np.zeros(steps, dtype='float64') # Renyi entropy
    neg = np.zeros(steps, dtype='float64') # logarithmic negativity
    mut = np.zeros(steps, dtype='float64') # mutual information using von-Neumann entropy
    mutr = np.zeros(steps, dtype='float64') # mutual information in terms of Renyi entropy
    
    for t in range(steps):
        # evolve over ALL links
        wave = unitary_mpi(wave, 4, l)     
        
        # measurement layer
        '''
        with this protocol, we need to double the measurement rate
        '''
        for i in range(l):
            wave = measure(wave, prob, i, l)
       
        result = ent(wave, n, l, l//2) # half-chain entanglement entropy
        # print(result[0])
        von[t] = result[0]
        renyi[t] = result[1]
        result = logneg(wave, n, partition) # logarithmic negativity according to preset partition
        neg[t] = result[0]
        mut[t] = result[1]
        mutr[t] = result[2]

    return np.array([von, renyi, neg, mut, mutr])

result = evo_parallel(time, psi, pro, L, 2, part)
np.savez('evo_L=%s_p=%s_t=%s'%(L, pro, time), ent=result[0], renyi=result[1], neg=result[2], mut=result[3], mutr=result[4])

end = timer()
print("Elapsed = %s" % (end - start))