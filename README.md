# Entanglement Dynamics: Introduction
Random unitary time evolution plus projective measurement in the one-dimensional quantum circuit model.

Starting from a product state without entanglement, by applying the randomly generated 2-qubit unitary gate between adjacent qubits, the bipartite entanglement entropy EE increases quickly in a linear fashion and thermalized in a volume law phases (EE ~ L).

If during the evolution, we measurement the state of those qubits in projectivly, e.g., a spin-z projection on some randomly selected site with probability p, there will be competitions between entanglement spreading and disentangling. The steady state of the system will be determined by these two factors. It is therefore natural to ask how the entanglement will evolve and the physical properties of the entanglement transition point.


Moreover, by partitioning the system into three parts, it is possible to calculate the mutual information 
and [negativity](https://en.wikipedia.org/wiki/Negativity_(quantum_mechanics)), which may provide more insight into the possible area-to-volume law entanglement phase transitions.

<!-- 
When the system size L is large, it is possible to make the approximation that only keeps a few O(L)
largest singular values in computing the entanglement entropy. This approximation works well even at the entanglement transition critical point where the entanglement scales as log(L).
 -->
# Dependencies:
1. Python 3 (tested with Python 3.7)
2. Jupyter lab (or notebook)
3. Numpy, Scipy, Numba (all lastest version from Conda)
4. [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) C++ library, [pybind11](https://github.com/pybind/pybind11) and [cppimport](https://github.com/tbenthompson/cppimport) if you want to try using C++ to speed up some matrix dot product.

GPU and MPI versions need Cupy and mpi4py and dependencies therein.

# TODO
~~GPU version for fast SVD and matrix dot.~~

The GPU version using [cupy](https://cupy.chainer.org/) is successfully tested on Kaggle platform (with NVidia K80). 

~~Large-scale distributed Kronecker product~~

Using mpi_evo.py for simulating relatively large system size (L\~ 24) with paralleled random unitary evolution.

Large-scale distributed SVD using MPI

# Refs:
The protocol used in the time evolution is the same as the following papers

[arXiv:1808.05953](https://arxiv.org/abs/1808.05953)

[arXiv:1901.08092](https://arxiv.org/abs/1901.08092)
