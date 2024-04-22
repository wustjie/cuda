#forward
template <typename T>                                                        
__global__ void ReluForwardKernel(const int N, const T* X, T* Y) { 
  const int i = blockIdx.x * blockdim.x + threadIdx.x;
  if (i < N) {
    Y[i] = (X[i] > T(0)) ? X[i] : T(0);
  }
}

#backward
template <typename T>
__global__ void ReluBackwardKernel(const int N, const T* dY, const T* Y, T* dX) {
  const int i = blockIdx.x * blockdim.x + threadIdx.x;
  if (i < N) {
    dX[i] = Y[i] > T(0) ? dY[i] : T(0);
  }
}
