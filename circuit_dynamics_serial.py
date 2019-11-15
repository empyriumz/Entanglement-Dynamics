import numpy as np 
from scipy import sparse
from scipy.stats import unitary_group
import random
from timeit import default_timer as timer 

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

# Pauli z-matrix
sz = np.array([[1, 0],[0, -1]])

class wavefunction:
    def __init__(self, len):
        p1 = np.ones(1)
        p2 = np.zeros(2**len-1,dtype='c16')
        # wavefunction in 1D form
        self.one_dim = np.concatenate((p1,p2),axis=0).T

    def ent(self, len, n, la): # von-Neumann and Renyi entanglement entropy
        lb = len-la
        # convert the wavefunction into a matrix for SVD
        temp = np.reshape(self.one_dim,(int(2**la),int(2**lb)))
        # SVD for entanglement entropy, only singular values are calculated
        sp = np.linalg.svd(temp, compute_uv=False)
        
        # chop small singular values to zero to avoid numerical instability
        tol = 1e-10
        sp[abs(sp) < tol] = 0.0
        # choose only non-zero values to avoid feeding to log function
        sp = sp[np.nonzero(sp)]
        el = sp**2
        # EE in log2 base
        return -np.dot(el,np.log2(el)),(1/(1-n))*np.log2(np.sum(sp**(2*n)))

    def logneg(self,n,la,lb,lc1,lc2):
        #region A
        ps = np.reshape(self.one_dim, (int(2**lc1),int(2**la),int(2**lc2),int(2**lb)))
        ps = np.moveaxis(ps,0,1)
        ps = np.reshape(ps,(int(2**la),int(2**(len-la))))
        # entanglement entropy in region A
        en = ent(ps, n, la) 
        # sa and sar stand for von-Neumann and Renyi entanglement entropies
        sa, sar = en[0], en[1]

    # time evolution consists of random unitaries + projective measurement
    def evo(steps, wave, prob):
        # initiate empty lists for keeping data
        von=np.zeros(steps, dtype='float64') # von-Neumann entropy
        renyi=np.zeros(steps, dtype='float64') # Renyi entropy
        neg=np.zeros(steps, dtype='float64') # logarithmic negativity
        mut=np.zeros(steps, dtype='float64') # mutual information using von-Neumann entropy
        mutr=np.zeros(steps, dtype='float64') # mutual information in terms of Renyi entropy
    
        for t in range(steps):
            # evolve over odd links
            u = unitary_group.rvs(4,size = len//2) # generate random U(4) matrix
            # combine the single unitary with rest of the identity matrix to make the complete unitary 
            # applying on the entire Hilbert space
            for i in range(len(u)):
                temp = sparse.kron(sparse.identity(2**(2*i)),u[i])
                un = sparse.kron(temp,sparse.identity(2**(L-2*i-2)))
                self.one_dim = un.dot(self.one_dim)

            # # measurement layer
            # self.one_dim = self.one_dim.measure(prob)

            # before evolve on even link, we need to rearrange indices first to accommodate the boundary condition PBC
            wave = np.reshape(self.one_dim,(2,int(2**(L-2)),2))
            # move the last site into the first one such that the unitaries can connect the 1st and the last site
            wave = np.moveaxis(wave,-1,0)
            self.one_dim = wave.flatten()
        
            # evolve over even links
            u = unitary_group.rvs(4,size=L//2) # generate another random unitary
            for i in range(len(u)):
                un = sparse.kron(sparse.identity(2**(2*i)),u[i])
                un = sparse.kron(un,sparse.identity(2**(L-2*i-2)))
                self.one_dim = un.dot(self.one_dim)

            #shift the index back to the original order after evolution
            wave = np.reshape(self.one_dim,(2,2,int(2**(L-2))))
            wave = np.moveaxis(wave,-1,0)
            self.one_dim = np.moveaxis(wave,-1,0).flatten()

            # # measurement layer
            # self.one_dim = self.one_dim.measure(prob)


            # calculate entanglement entropies with bi-partition
            result = ent(wave,2,L//2)
            von[t] = result[0]
            renyi[t] = result[1]
            # calculate logarithmic negativity and mutual information with tri-partition
            result = logneg(wave,2,la,lb,lc1,lc2)
            neg[t] = result[0]
            mut[t] = result[1]
            mutr[t] = result[2]
        
    return von, renyi, neg, mut, mutr


psi = wavefunction(L)
print(psi.ent(L, 2, 3))




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
def measure(wave,prob):
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
            pdown=wave.conjugate().T.dot(pdown1)
            pdown=1-pup

            # probility of the measurement is determined by the projection 
            out = np.random.choice([0,1],1,p=[pup, 1-pup])

            # if the measurement projects the spin onto z-up state
            if out[0]==0:
                wave=(1/np.sqrt(pup))*pup1 # normalization of wavefunction
            else:
                wave=(1/np.sqrt(pdown))*pdown1
    return wave

# time evolution consists of random unitaries + projective measurement
def evo(steps, wave, prob):
    # initiate empty lists for keeping data
    von=np.zeros(steps, dtype='float64') # von-Neumann entropy
    renyi=np.zeros(steps, dtype='float64') # Renyi entropy
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
        # calculate logarithmic negativity and mutual information with tri-partition
        result=logneg(wave,2,la,lb,lc1,lc2)
        neg[t]=result[0]
        mut[t]=result[1]
        mutr[t]=result[2]
        
    return von, renyi, neg, mut, mutr


# result = evo(time, psi, pro)
# np.savez_compressed('evolution_L=%s_t=%s_p=%s'%(L, time, pro), von=result[0], renyi=result[1],neg=result[2], mutual=result[3], mutualrenyi=result[4])


end = timer()
print("Elapsed = %s" % (end - start))