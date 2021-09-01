import numpy as np
import pyopencl as cl
import pyopencl.array
import pyamg
from time import time

ntests = 10   # how many timing tests to run?

# load a matrix
data = pyamg.gallery.load_example('unit_square')
A = data['A'].tocsr()
n_row, n_col = A.shape
x = np.random.rand(n_col)
y = np.zeros(n_row)
c = np.array([1.0])

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

c_dev = cl.array.to_device(queue, c)
x_dev = cl.array.to_device(queue, x)
y_dev = cl.array.to_device(queue, y)
Ap_dev = cl.array.to_device(queue, A.indptr)
Aj_dev = cl.array.to_device(queue, A.indices)
Ax_dev = cl.array.to_device(queue, A.data)

prg = cl.Program(ctx, """
// y = A * x
// A = Ap (row pointer)
//     Aj (col indices)
//     Ax (data)
 __kernel void
spmv_csr_scalar(__global const int * Ap,
                __global const int * Aj,
                __global const double * Ax,
                __global const double * x,
                __global double * y,
                const unsigned int n_row)
{
    int i = get_global_id(0);

    double sum=0;
    for (int jj = Ap[i]; jj < Ap[i+1]; jj++)
    {
        sum += Ax[jj] * x[Aj[jj]];
    }
    y[i] = sum;
}
"""
        ).build()

times = []
for k in range(ntests+3): # warmup = 3
    knl = prg.spmv_csr_scalar(queue, y.shape, None,
                              Ap_dev.data, Aj_dev.data, Ax_dev.data,
                              x_dev.data, y_dev.data,
                              np.int32(n_row))
    knl.wait()
    time_knl = 1e-9 * (knl.profile.end - knl.profile.start)
    if k >= 3:
        # let the GPU warm up 3x
        i = k-3
        times.append(time_knl)
y_dev.get(ary=y)

print("mean time:   ", np.mean(times))
print("mean gflops: ", A.nnz/1e9 / np.mean(times))
print("check: ", np.linalg.norm(A @ x - y))
