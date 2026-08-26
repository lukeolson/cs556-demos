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
#kernel_file = 'spmv_csr_vector.cl'

wgsize = 32   # work group size (power of 2)

# define the precision as real_t in the .cl kernels
header = """
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
typedef %s real_t;
#define WGSIZE %d
""" % (cl_type, wgsize)

with open(kernel_file) as f:
    prg = cl.Program(ctx, header + f.read()).build()

kernel_name = kernel_file.replace('.cl', '')
knl = cl.Kernel(prg, kernel_name)

# scalar: one work item per row
# vector: one work group per row
if kernel_name.endswith('_vector'):
    max_wg = knl.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, ctx.devices[0])
    if wgsize > max_wg:
        raise RuntimeError(f'max work group size is {max_wg}')
    gsize, lsize = (n_row * wgsize,), (wgsize,)
else:
    gsize, lsize = (n_row,), None

times = []
for k in range(ntests+3): # warmup = 3
    evt = knl(queue, gsize, lsize,
              Ap_dev.data, Aj_dev.data, Ax_dev.data,
              x_dev.data, y_dev.data,
              np.int32(n_row))
    evt.wait()
    time_knl = 1e-9 * (evt.profile.end - evt.profile.start)
    if k >= 3:
        # let the GPU warm up 3x
        times.append(time_knl)
y_dev.get(ary=y)

print("kernel:      ", kernel_name)
print("precision:   ", cl_type)
print("mean time:   ", np.mean(times))
# 2 flops per nonzero: one multiply and one add
print("mean gflops: ", 2 * A.nnz/1e9 / np.mean(times))
print("check: ", np.linalg.norm(A @ x - y))
