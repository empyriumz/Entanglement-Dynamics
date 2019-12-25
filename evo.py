from circuit_dynamics_init import *
from pybind_circuit import unitary_cxx_parallel, unitary_cxx, unitary_conventional_cxx
from timeit import default_timer as timer

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

# pure python version of evolution
# suitable for small size L<=18 
def evo(steps, wave, prob, k = 4, l = L, n = 2, partition = part):
    von = np.zeros(steps, dtype='float64') # von-Neumann entropy
    renyi = np.zeros(steps, dtype='float64') # Renyi entropy
    neg = np.zeros(steps, dtype='float64') # logarithmic negativity
    mut = np.zeros(steps, dtype='float64') # mutual information using von-Neumann entropy
    mutr = np.zeros(steps, dtype='float64') # mutual information in terms of Renyi entropy
    
    for t in range(steps):
        # evolve over All links
        wave = unitary(wave, k, l)   
        
        # measurement layer
        '''
        with this protocol, we need to double the measurement rate
        '''
        for j in range(l):
            wave = measure(wave, prob, j, l)
       
        result = ent(wave, n, l//2, l) # half-chain entanglement entropy
        von[t] = result[0]
        renyi[t] = result[1]
        result = logneg(wave, n, partition)
        neg[t] = result[0]
        mut[t] = result[1]
        mutr[t] = result[2]

    return np.array([von, renyi, neg , mut, mutr])

# pure python version with original evolution scheme
def evo_original(steps, wave, prob, l = L, n = 2, partition = part):
    von = np.zeros(steps, dtype='float64') # von-Neumann entropy
    renyi = np.zeros(steps, dtype='float64') # Renyi entropy
    neg = np.zeros(steps, dtype='float64') # logarithmic negativity
    mut = np.zeros(steps, dtype='float64') # mutual information using von-Neumann entropy
    mutr = np.zeros(steps, dtype='float64') # mutual information in terms of Renyi entropy
    
    for t in range(steps):
        # evolve over odd links
        for i in range(l//2):
            wave = unitary_conventional(wave, i, l)     
        
        # measurement layer
        for i in range(l):
            wave = measure(wave, prob, i, l)

        # before evolve on even link, we need to rearrange indices first to accommodate the boundary condition PBC
        wave = np.reshape(wave,(2, 2**(l-2),2))
        # move the last site into the first one such that the unitaries can connect the 1st and the last site
        wave = np.moveaxis(wave,-1,0)
        wave = wave.flatten()
        
        # evolve over even links
        for i in range(l//2):
            wave = unitary_conventional(wave, i, l)  

        #shift the index back to the original order after evolution
        wave = np.reshape(wave,(2, 2, 2**(l-2)))
        wave = np.moveaxis(wave,-1,0)
        wave = np.moveaxis(wave,-1,0).flatten()

        #measurement layer
        for i in range(l):
            wave = measure(wave, prob, i, l)
       
        result = ent(wave, n, l//2, l)
        von[t] = result[0]
        renyi[t] = result[1]
        result = logneg(wave, n, partition)
        neg[t] = result[0]
        mut[t] = result[1]
        mutr[t] = result[2]

    return np.array([von, renyi, neg , mut, mutr])

# replacing dot product with c++ version of evolution
# slowest option
def evo_cxx(steps, wave, prob, k = 4, l = L, n = 2, partition = part):
    von = np.zeros(steps, dtype='float64') # von-Neumann entropy
    renyi = np.zeros(steps, dtype='float64') # Renyi entropy
    neg = np.zeros(steps, dtype='float64') # logarithmic negativity
    mut = np.zeros(steps, dtype='float64') # mutual information using von-Neumann entropy
    mutr = np.zeros(steps, dtype='float64') # mutual information in terms of Renyi entropy
    '''
    only the splitted dot product is replaced by c++ module
    '''
    for t in range(steps):
        # evolve over All links
        wave = unitary_cxx(wave, k, l)   
        
        # measurement layer
        '''
        with this protocol, we need to double the measurement rate
        '''
        for j in range(l):
            wave = measure(wave, prob, j, l)
       
        result = ent_approx(wave, n, l, l//2) # half-chain entanglement entropy
        von[t] = result[0]
        renyi[t] = result[1]
        result = logneg(wave, n, partition)
        neg[t] = result[0]
        mut[t] = result[1]
        mutr[t] = result[2]

    return np.array([von, renyi, neg , mut, mutr])

# replacing dot product with c++ 
def evo_original_cxx(steps, wave, prob, l = L, n = 2, partition = part):
    von = np.zeros(steps, dtype='float64') # von-Neumann entropy
    renyi = np.zeros(steps, dtype='float64') # Renyi entropy
    neg = np.zeros(steps, dtype='float64') # logarithmic negativity
    mut = np.zeros(steps, dtype='float64') # mutual information using von-Neumann entropy
    mutr = np.zeros(steps, dtype='float64') # mutual information in terms of Renyi entropy
    
    for t in range(steps):
        # evolve over odd links
        for i in range(l//2):
            wave = unitary_conventional_cxx(wave, i, l)     
        
        # measurement layer
        for i in range(l):
            wave = measure(wave, prob, i, l)

        # before evolve on even link, we need to rearrange indices first to accommodate the boundary condition PBC
        wave = np.reshape(wave,(2, 2**(l-2),2))
        # move the last site into the first one such that the unitaries can connect the 1st and the last site
        wave = np.moveaxis(wave,-1,0)
        wave = wave.flatten()
        
        # evolve over even links
        for i in range(l//2):
            wave = unitary_conventional_cxx(wave, i, l)  

        #shift the index back to the original order after evolution
        wave = np.reshape(wave,(2, 2, 2**(l-2)))
        wave = np.moveaxis(wave,-1,0)
        wave = np.moveaxis(wave,-1,0).flatten()

        #measurement layer
        for i in range(l):
            wave = measure(wave, prob, i, l)
       
        result = ent(wave, n, l//2, l)
        von[t] = result[0]
        renyi[t] = result[1]
        result = logneg(wave, n, partition)
        neg[t] = result[0]
        mut[t] = result[1]
        mutr[t] = result[2]

    return np.array([von, renyi, neg , mut, mutr])

# c++ binding accelerated unitary evolution
def evo_parallel(steps, wave, prob, k = 4, l = L, n = 2, partition = part):
    von = np.zeros(steps, dtype='float64') # von-Neumann entropy
    renyi = np.zeros(steps, dtype='float64') # Renyi entropy
    neg = np.zeros(steps, dtype='float64') # logarithmic negativity
    mut = np.zeros(steps, dtype='float64') # mutual information using von-Neumann entropy
    mutr = np.zeros(steps, dtype='float64') # mutual information in terms of Renyi entropy
    
    for t in range(steps):
        # evolve over All links
        wave = unitary_cxx_parallel(wave, k, l)     
        
        # measurement layer
        '''
        with this protocol, we need to double the measurement rate
        '''
        for j in range(l):
            wave = measure(wave, prob, j, l)
       
        result = ent(wave, n, l//2, l) # half-chain entanglement entropy
        von[t] = result[0]
        renyi[t] = result[1]
        result = logneg(wave, n, partition)
        neg[t] = result[0]
        mut[t] = result[1]
        mutr[t] = result[2]

    return np.array([von, renyi, neg , mut, mutr])

# benchmark for different evolution schemes
# start = timer()
# result = evo_original(time, psi, pro)
# end = timer()
# print("Elapsed = %s" % (end - start))

start = timer()
result = evo_original_cxx(time, psi, pro)
end = timer()
print("Elapsed = %s" % (end - start))


start = timer()
result = evo_parallel(time, psi, 2*pro, 6)
end = timer()
print("Elapsed = %s" % (end - start))

start = timer()
result = evo_parallel(time, psi, 2*pro, 4)
end = timer()
print("Elapsed = %s" % (end - start))
np.savez('dynamics_L=%s_p=%s_t=%s'%(L, pro, time), ent=result[0], renyi=result[1], neg=result[2], mut=result[3], mutr=result[4])