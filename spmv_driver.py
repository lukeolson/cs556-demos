import numpy as np
import pyopencl as cl
import pyopencl.array
import pyamg
from time import time

ntests = 10   # how many timing tests to run?

precision, cl_type = np.float32, 'float'
#precision, cl_type = np.float64, 'double'

# load a matrix
data = pyamg.gallery.load_example('unit_square')
A = data['A'].tocsr().astype(precision)
n_row, n_col = A.shape
x = np.random.rand(n_col).astype(precision)
y = np.zeros(n_row, dtype=precision)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

if precision == np.float64:
    for dev in ctx.devices:
        if dev.double_fp_config == 0:
            raise RuntimeError(f'device does not support double precision')

x_dev = cl.array.to_device(queue, x)
y_dev = cl.array.to_device(queue, y)
Ap_dev = cl.array.to_device(queue, A.indptr)
Aj_dev = cl.array.to_device(queue, A.indices)
Ax_dev = cl.array.to_device(queue, A.data)

kernel_file = 'spmv_csr_scalar.cl'

# define the precision as real_t in the .cl kernels
header = """
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
typedef %s real_t;
""" % cl_type

with open(kernel_file) as f:
    prg = cl.Program(ctx, header + f.read()).build()

knl = cl.Kernel(prg, kernel_file.replace('.cl', ''))

times = []
for k in range(ntests+3): # warmup = 3
    evt = knl(queue, y.shape, None,
              Ap_dev.data, Aj_dev.data, Ax_dev.data,
              x_dev.data, y_dev.data,
              np.int32(n_row))
    evt.wait()
    time_knl = 1e-9 * (evt.profile.end - evt.profile.start)
    if k >= 3:
        # let the GPU warm up 3x
        times.append(time_knl)
y_dev.get(ary=y)

print("precision:   ", cl_type)
print("mean time:   ", np.mean(times))
# 2 flops per nonzero: one multiply and one add
print("mean gflops: ", 2 * A.nnz/1e9 / np.mean(times))
print("check: ", np.linalg.norm(A @ x - y))
