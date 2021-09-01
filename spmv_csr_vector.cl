// y = A * x
// A = Ap (row pointer)
//     Aj (col indices)
//     Ax (data)
 __kernel void
spmv_csr_vector(__global const int * Ap,
                __global const int * Aj,
                __global const double * Ax,
                __global const double * x,
                __global double * y,
                const unsigned int n_row)
{
    int wid = get_local_id(0); 
    int wgsize = get_local_size(0);
    int i = get_group_id(0);

    __local volatile double wgsum[128];
    wgsum[wid] = 0;
    
    if (i < n_row ) {
        double sum=0;
        for (int jj = Ap[i] + wid; jj < Ap[i+1]; jj+=wgsize)
        {
            sum += Ax[jj] * x[Aj[jj]];
        }

        wgsum[wid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        // reduction over the work group
        for (uint sz=wgsize/2; sz>0; sz /= 2){
            if (wid < sz)
            {
                wgsum[wid] += wgsum[wid + sz];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (wid == 0){
            y[i] = wgsum[0];
        }
    }
}
