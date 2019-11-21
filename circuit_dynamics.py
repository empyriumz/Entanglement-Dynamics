import numpy as np 
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import unitary_group
import random
from timeit import default_timer as timer 
from numba import jit

# keep track of running time
start = timer()

# reading parameters from file
para = open('para_haar.txt', 'r')
para = para.readlines()
# the paramters are system size, measurement probability and discrete time steps
L, pro, time = int(para[0]), float(para[1]), int(para[2])

# system partition
# with PBC, we partition system into 4 parts where a and b separated by c1 and c2
# c1 and c2 are effectively connected, so the system is composed of A, B and C
lc1,la,lb = np.floor(L/8), np.floor(L/4), np.floor(L/4)
lc2=L-lc1-la-lb

# initializing wavefunctions
p1=np.ones(1)
p2=np.zeros(2**L-1,dtype='c16')
# a product state with all spins align up
psi=np.concatenate((p1,p2),axis=0).T
# Pauli z-matrix
sz = np.array([[1, 0],[0, -1]])

@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def kron_raw(d_a, r_a, c_a, d_b, r_b, c_b):
    
    # nz = #s of stored values, including explicit zeros

    # use COO
    
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
    #print(row, B.row, B.col)
    
    # this can be verified from the definitions of kronecker product, see Wiki for the formula
    row += r_b
    col += c_b
    row, col = row.reshape(-1), col.reshape(-1)
    # compute block entries
    data = data.reshape(-1,nz) * d_b
    data = data.reshape(-1)

    return data, row, col



# von-Neumann and Renyi entanglement entropy
def ent(wave,n,la):
    lb=L-la
    # convert the wavefunction into a matrix for SVD
    temp=np.reshape(wave,(int(2**la),int(2**lb)))
    # SVD for entanglement entropy, only singular values calculated
    sp=np.linalg.svd(temp, compute_uv=False)
    tol=1e-10
    # chop small singular values to zero to avoid numerical instability
    sp[abs(sp) < tol] = 0.0
    # choose only non-zero values to avoid feeding to log function
    sp=sp[np.nonzero(sp)]
    el=sp**2
    return -np.dot(el,np.log2(el)),(1/(1-n))*np.log2(np.sum(sp**(2*n)))

def ent_approx(wave, n, la): 
	# approximated von-Neumann and Renyi entanglement entropy 
	# by keeping L largest singluar values
	l = L
	lb = l - la
    # convert the wavefunction into a matrix for SVD
	temp = np.reshape(wave,(int(2**la),int(2**lb)))
	# (*The singular values can be obtained from Sqrt[Eigenvalues[ConjugateTranspose[m].m]]. *)
	temp = sparse.csr_matrix(np.dot(temp, temp.conj().T))
	sp = np.array(eigsh(temp, k = l, which='LM',return_eigenvectors=False))
	
	# chop small singular values to zero to avoid numerical instability
	tol = 1e-10
	sp[abs(sp) < tol] = 0.0
	# retain only non-zero values to avoid feeding to log function
	el = sp[sp != 0]
	# el = sp**2
	# EE in log2 base
	return -np.dot(el,np.log2(el)), (1/(1-n))*np.log2(np.sum(sp**(n)))

# logarithmic negativity and mutual information
def logneg(wave,n,la,lb,lc1,lc2):
	tol=1e-10
    #region A
	ps=np.reshape(wave, (int(2**lc1),int(2**la),int(2**lc2),int(2**lb)))
	ps=np.moveaxis(ps,0,1)
	ps=np.reshape(ps,(int(2**la),int(2**(L-la))))
	# entanglement entropy in region A
	en = ent(ps, n, la)	
	# sa and sar stand for von-Neumann and Renyi entanglement entropies
	sa, sar = en[0], en[1]
	

	#region B
	ps=np.reshape(wave, (int(2**(L-lb)),int(2**lb)))
	en = ent(ps, n, L-lb)
	sb, sbr = en[0], en[1]

	#region C
	# since C composed of c1 and c2, we need to re-arrange the index to combine c1 and c2 into
	# a connected region
	ps=np.reshape(wave, (int(2**lc1),int(2**la),int(2**lc2),int(2**lb)))
	ps=np.moveaxis(ps,1,2)
	ps=np.reshape(ps,(int(2**(lc1+lc2)),int(2**(la+lb))))
	en = ent(ps, n, lc1+lc2)
	sc, scr = en[0], en[1]
	
	# log(negativity)
	rab=np.dot(ps.T,np.conjugate(ps)) #reduced density matrix by tracing out C
	# reshape the reduced density matrix to have 4 indices to facilitate partial transpose
	rab=np.reshape(rab,(int(2**la),int(2**lb),int(2**la),int(2**lb)))

	# partial transpose on A
	pab=np.moveaxis(rab,0,2)
	# rearrange indices to make pab into a matrix
	pab=pab.reshape(int(2**(la+lb)),int(2**(la+lb)))
	# SVD of partial transposed density matrix
	sp=np.linalg.svd(pab, compute_uv=False)
	# definition of logarithmic negativity
	logn=np.log2(np.sum(sp))

	# returns logarithmic negativity and two mutual information
	return abs(logn), sa+sb-sc, sar+sbr-scr

# projective single-site measurement in z-basis
def measure_slow(wave,prob):
	'''
	the z-projection operator for each location is constructed
	in a straightforward manner by making tensor product between s_z and 
	identity matrix
	'''
	# there are two possible outcomes
	choice = [0, 1]
	for n in range(L):
		op=np.random.choice(choice, 1, p=[1-prob, prob]) #determine if to measure on the site
		# if the measurement is chosen at given position
		if op[0]==1:
			# projection operator for spin up in z-basis
			up=sparse.kron(sparse.identity(2**n),sz)
			up=sparse.kron(up,sparse.identity(2**(L-n-1)))
			up=0.5*(up+sparse.identity(2**L)) 

			# projection operator for spin down
			down=sparse.kron(sparse.identity(2**n),sz)
			down=sparse.kron(down,sparse.identity(2**(L-n-1)))
			down=0.5*(sparse.identity(2**L)-down) 

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
def measure(wave, prob, pos):
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
        temp = np.ones(2**(L-pos-1))
        pz = np.concatenate((temp,-temp))
        # repeat the pattern for 2**pos times        
        pz = np.tile(pz, 2**pos)
        '''
        we can construct the diagonal projection matrix as follows 
        pz = sparse.diags(np.tile(pz, 2**pos))
        but it's unnecessary since we can avoid the dot product between a diagonal matrix and 
        a vector by direct list multiplication
        
        alternatively, this can be done with the more conventional method
        pz = sparse.kron(sparse.identity(2**pos), sz, format="csr")
        pz = sparse.kron(pz,sparse.identity(2**(L-pos-1)), format="csr")
        but the kronecker product is slow in scipy
        '''
        

        # projection of wavefunction
        '''
        the spin-up/dowm projection operator is given by $1/2(I \pm \sigma_z)$
        we compute \sigma_z dot \psi part only since the identity part is trivial.
        This is also for avoiding the plus operation between 
        two large sparse matrices which is slow numerically
        '''
        temp = pz*wave
        '''
        instead we only need to compute the plus and minus between vectors
        '''
        pup1 = 0.5*(wave + temp)
        pdown1 = 0.5*(wave - temp)
        # expectation values
        #temp = (wave.conjugate().T).dot(temp)
        temp = np.vdot(wave, temp)
        pup = 0.5 + 0.5*np.asscalar(temp.real)
        pdown = 1 - pup

        # projection of wavefunction into z-down state
        #pdown1=down.dot(wave)
        #pdown=wave.conjugate().T.dot(pdown1)
        #pdown=np.asscalar(pdown.real)
        
        '''
        in case the wavefunction is close to product state, the measurement 
        might yield pup=1 or pdown=1. 
        To avoid possible numerical errors where pup>1, we manually set the probability 
        to be 1 or 0.
        '''
        if abs(pup-1)<1e-8: 
            pup = 1.0
            pdown = 0.0
            wave = pup1
        elif abs(pup)<1e-8:
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
                wave = (1/np.sqrt(pup))*pup1 # normalization of wavefunction
            else:
                wave = (1/np.sqrt(pdown))*pdown1
            # print(pup, pdown)
            
    return wave

# time evolution consists of random unitaries + projective measurement
def evo(steps, wave, prob):
	# initiate empty lists for keeping data
	von=np.zeros(steps, dtype='float64') # von-Neumann entropy
	renyi=np.zeros(steps, dtype='float64') # Renyi entropy
	von2 = np.zeros(steps, dtype='float64') # von-Neumann entropy
	renyi2 = np.zeros(steps, dtype='float64') # Renyi entropy
	neg=np.zeros(steps, dtype='float64') # logarithmic negativity
	mut=np.zeros(steps, dtype='float64') # mutual information using von-Neumann entropy
	mutr=np.zeros(steps, dtype='float64') # mutual information in terms of Renyi entropy
	
	for t in range(steps):
		# evolve over odd links
		u=unitary_group.rvs(4,size=L//2) # generate random U(4) matrix
		# combine the single unitary with rest of the identity matrix to make the complete unitary 
		# applying on the entire Hilbert space
		for i in range(len(u)):
			temp=sparse.kron(sparse.identity(2**(2*i)),u[i])
			un=sparse.kron(temp,sparse.identity(2**(L-2*i-2)))
			wave=un.dot(wave)

		# measurement layer
		wave=measure(wave,prob)

		# before evolve on even link, we need to rearrange indices first to accommodate the boundary condition PBC
		wave=np.reshape(wave,(2,int(2**(L-2)),2))
		# move the last site into the first one such that the unitaries can connect the 1st and the last site
		wave=np.moveaxis(wave,-1,0)
		wave=wave.flatten()
		
		# evolve over even links
		u=unitary_group.rvs(4,size=L//2)
		for i in range(len(u)):
			un=sparse.kron(sparse.identity(2**(2*i)),u[i])
			un=sparse.kron(un,sparse.identity(2**(L-2*i-2)))
			wave=un.dot(wave)

		#shift the index back to the original order after evolution
		wave=np.reshape(wave,(2,2,int(2**(L-2))))
		wave=np.moveaxis(wave,-1,0)
		wave=np.moveaxis(wave,-1,0).flatten()

		#measurement layer
		wave=measure(wave,prob)

		# calculate entanglement entropies with bi-partition
		result=ent(wave,2,L//2)
		von[t]=result[0]
		renyi[t]=result[1]

		result=ent2(wave,2,L//2)
		von2[t]=result[0]
		renyi2[t]=result[1]
		# calculate logarithmic negativity and mutual information with tri-partition
		# result=logneg(wave,2,la,lb,lc1,lc2)
		# neg[t]=result[0]
		# mut[t]=result[1]
		# mutr[t]=result[2]
		
	# return von, renyi, neg, mut, mutr, wave
	return von, renyi, von2, renyi2

# compile jit
kron_raw(d_a, r_a, c_a, d_b, r_b, c_b);

result = evo(time, psi, pro)
print(result[0]-result[2])
print(result[1]-result[3])
#np.savez_compressed('evolution_L=%s_t=%s_p=%s'%(L, time, pro), von=result[0], renyi=result[1],neg=result[2], mutual=result[3], mutualrenyi=result[4])


end = timer()
print("Elapsed = %s" % (end - start))