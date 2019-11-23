# Entanglement-Dynamics
Random unitary time evolution plus projective measurement in the one-dimensional quantum circuit model.

By partitioning the system into three parts, it is possible to calculate the mutual information 
and negativity, which provides more insight into the possible area-to-volume law entanglement phase transitions.

When the system size L is large, it is possible to make the approximation that only keeps a few O(L)
largest singular values in computing the entanglement entropy. This approximation works well even at the entanglement transition critical point where the entanglement scales as log(L).

# TODO
~~GPU version using [cupy](https://cupy.chainer.org/) for fast SVD and matrix dot.~~

The GPU version is successfully tested on Kaggle platform (with NVidia K80).


# Refs:
The protocol used in the time evolution is the same as the following papers

[arXiv:1808.05953](https://arxiv.org/abs/1808.05953)

[arXiv:1901.08092](https://arxiv.org/abs/1901.08092)
