import numpy as np 
from scipy import sparse
from scipy.stats import unitary_group
import random
from timeit import default_timer as timer 

start = timer()

#reading parameters from file
para = open('para_haar.txt', 'r')
para = para.readlines()
L, pro, time = int(para[0]), float(para[1]), int(para[2])

#system partition
lc1,la,lb = np.floor(L/8), np.floor(L/4), np.floor(L/4)
lc2=L-lc1-la-lb

#initializing wavefunctions
p1=np.ones(1)
p2=np.zeros(2**L-1,dtype='c16')
psi=np.concatenate((p1,p2),axis=0).T
sz = np.array([[1, 0],[0, -1]])

def list(list1):
    return str(list1).replace('[','').replace(']','').replace('+0j','').replace('(','').replace(')','')

#von-Neumann entanglement entropy and Renyi entropy
def ent(wave,n,la):
    lb=L-la
    temp=np.reshape(wave,(int(2**la),int(2**lb)))
    sp=np.linalg.svd(temp, compute_uv=False)
    tol=1e-10
    #chop small singular values to zero to avoid numerical instability
    sp[abs(sp) < tol] = 0.0
    #choose only non-zero values to avoid feeding to log
    sp=sp[np.nonzero(sp)]
    el=sp**2
    return -np.dot(el,np.log2(el)),(1/(1-n))*np.log2(np.sum(sp**(2*n)))

#logarithmic negativity and mutual information
def logneg(wave,n,la,lb,lc1,lc2):
	tol=1e-10
    #region A
	ps=np.reshape(wave, (int(2**lc1),int(2**la),int(2**lc2),int(2**lb)))
	ps=np.moveaxis(ps,0,1)
	ps=np.reshape(ps,(int(2**la),int(2**(L-la))))
	sp=np.linalg.svd(ps, compute_uv=False)
	sp[abs(sp) < tol] = 0.0
	sp=sp[np.nonzero(sp)]
	el=sp**2
	sa=-np.dot(el,np.log2(el))
	sar=(1/(1-n))*np.log2(np.sum(sp**(2*n)))

	#region B
	ps=np.reshape(wave, (int(2**(L-lb)),int(2**lb)))
	sp=np.linalg.svd(ps, compute_uv=False)
	sp[abs(sp) < tol] = 0.0
	sp=sp[np.nonzero(sp)]
	el=sp**2
	sb=-np.dot(el,np.log2(el))
	sbr=(1/(1-n))*np.log2(np.sum(sp**(2*n)))

	#region C
	ps=np.reshape(wave, (int(2**lc1),int(2**la),int(2**lc2),int(2**lb)))
	ps=np.moveaxis(ps,1,2)
	ps=np.reshape(ps,(int(2**(lc1+lc2)),int(2**(la+lb))))
	sp=np.linalg.svd(ps, compute_uv=False)
	sp[abs(sp) < tol] = 0.0
	sp=sp[np.nonzero(sp)]
	el=sp**2
	sc=-np.dot(el,np.log2(el))
	scr=(1/(1-n))*np.log2(np.sum(sp**(2*n)))

	#log(negativity)
	rab=np.dot(ps.T,np.conjugate(ps)) #reduced density matrix by tracing out C
	rab=np.reshape(rab,(int(2**la),int(2**lb),int(2**la),int(2**lb)))
	#partial transpose on A
	pab=np.moveaxis(rab,0,2)
	pab=pab.reshape(int(2**(la+lb)),int(2**(la+lb)))
	sp=np.linalg.svd(pab, compute_uv=False)
	logn=np.log2(np.sum(sp))
	#logn[abs(logn) < tol] = 0.0
	#logn.real[abs(logn.real) < tol] = 0.0
	return abs(logn), sa+sb-sc, sar+sbr-scr

#projective single-site measurement in z-basis
def measure(wave,prob):
	choice = [0, 1]
	for n in range(L):
		op=np.random.choice(choice, 1, p=[1-prob, prob]) #determine if to measure on the site
		#if the measurement is chosen at given position
		if op[0]==1:
			up=sparse.kron(sparse.identity(2**n),sz)
			up=sparse.kron(up,sparse.identity(2**(L-n-1)))
			up=0.5*(up+sparse.identity(2**L)) #projection operator for spin up
			down=sparse.kron(sparse.identity(2**n),sz)
			down=sparse.kron(down,sparse.identity(2**(L-n-1)))
			down=0.5*(sparse.identity(2**L)-down) #projection operator for spin down
			pup1=up.dot(wave)
			pup=(wave.conjugate().T).dot(pup1)
			pup=np.asscalar(pup.real)
			pdown1=down.dot(wave)
			pdown=wave.conjugate().T.dot(pdown1)
			pdown=1-pup
			out = np.random.choice([0,1],1,p=[pup, 1-pup])
			if out[0]==0:
				wave=(1/np.sqrt(pup))*pup1 #normalization of wavefunction
			else:
				wave=(1/np.sqrt(pdown))*pdown1
	return wave

def evo(steps, wave, prob):
	von=np.zeros(steps, dtype='float64')
	renyi=np.zeros(steps, dtype='float64')
	neg=np.zeros(steps, dtype='float64')
	mut=np.zeros(steps, dtype='float64')
	mutr=np.zeros(steps, dtype='float64')
	for t in range(steps):
		#evolve over odd links
		u=unitary_group.rvs(4,size=L//2)
		for i in range(len(u)):
			temp=sparse.kron(sparse.identity(2**(2*i)),u[i])
			un=sparse.kron(temp,sparse.identity(2**(L-2*i-2)))
			wave=un.dot(wave)

		#measurement layer
		wave=measure(wave,prob)

		#before evolve on even link, we need to rearrange indices first
		wave=np.reshape(wave,(2,int(2**(L-2)),2))
		wave=np.moveaxis(wave,-1,0)
		wave=wave.flatten()
		
		#evolve over even links
		u=unitary_group.rvs(4,size=L//2)
		for i in range(len(u)):
			un=sparse.kron(sparse.identity(2**(2*i)),u[i])
			un=sparse.kron(un,sparse.identity(2**(L-2*i-2)))
			wave=un.dot(wave)

		#shift the index back to original order after evolving
		wave=np.reshape(wave,(2,2,int(2**(L-2))))
		wave=np.moveaxis(wave,-1,0)
		wave=np.moveaxis(wave,-1,0).flatten()

		#measurement layer
		wave=measure(wave,prob)

		#calculate EE
		result=ent(wave,2,L//2)
		von[t]=result[0]
		renyi[t]=result[1]
		# result=logneg(wave,2,la,lb,lc1,lc2)
		# neg[t]=result[0]
		# mut[t]=result[1]
		# mutr[t]=result[2]
		
	return von, renyi

import os
num=os.path.basename(__file__)[-4]
result=evo(time, psi, pro)
result[0].tofile('%see_L=%s_p=%s_t=%s.dat'%(num,L,pro,time))
result[1].tofile('%srenyi_L=%s_p=%s_t=%s.dat'%(num,L,pro,time))
# result[2].tofile('%sneg_L=%s_p=%s_t=%s.dat'%(num,L,pro,time))

end = timer()
print("Elapsed = %s" % (end - start))