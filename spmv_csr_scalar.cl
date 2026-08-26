// y = A * x
// A = Ap (row pointer)
//     Aj (col indices)
//     Ax (data)
 __kernel void
spmv_csr_scalar(__global const int * Ap,
                __global const int * Aj,
                __global const real_t * Ax,
                __global const real_t * x,
                __global real_t * y,
                const unsigned int n_row)
{
    int i = get_global_id(0);

    real_t sum=0;
    for (int jj = Ap[i]; jj < Ap[i+1]; jj++)
    {
        sum += Ax[jj] * x[Aj[jj]];
    }
    y[i] = sum;
}
