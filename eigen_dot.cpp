<%
cfg['compiler_args'] = ['-std=c++14','-fopenmp']
cfg['linker_args'] = ['-fopenmp']
cfg['include_dirs'] = ['/home/eigen-3.3.7']
setup_pybind11(cfg)
%>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <omp.h>
#define BATCHSIZE 8 
namespace py = pybind11;
using namespace Eigen;

typedef SparseMatrix<std::complex<double>>  SpMat; 


VectorXcd dot(int i, int l, SpMat un, VectorXcd wave)
{
 py::gil_scoped_acquire acquire; /* Acquire GIL before calling Python code */
 initParallel(); // not required after eigen 3.3 and c++11 compiler
 setNbThreads(8);
 int k;
 int chunk = pow(2, l-2*i); // dimensions of the incoming unitary sparse matrix
 int dim = pow(2, l);
 int split = pow(2, 2*i); // # of splitted wavefunction
 int batch = BATCHSIZE; // specify the batch size for each threads
 VectorXcd temp(dim); // temp vector to keep tract of mat-vec multiplication
 temp = VectorXcd::Zero(dim);
 // paralleling using openmp
 #pragma omp parallel shared(un, wave) private(k)
 {
 // since each mat-vec dot product is totally independent, we can safely parallel the loop 
  #pragma omp for schedule(dynamic,batch)
  for (k = 0; k < split; k++){
    temp.segment(k*chunk, chunk) = un * wave.segment(k*chunk, chunk);
  }
}

 return temp;

}

VectorXcd dot_simple(SpMat un, VectorXcd wave)
{
 return un*wave;
}

// PYBIND11_MODULE(dot_simple, m) {   
//     m.def("dot", &dot);  
//  }

PYBIND11_MODULE(eigen_dot, m) {   
    //Release GIL before calling into C++ code 
    m.def("dot", &dot, py::call_guard<py::gil_scoped_release>());
    m.def("dot_simple", &dot_simple, py::call_guard<py::gil_scoped_release>());  
 }