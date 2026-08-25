#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

const int THREADS_PER_BLOCK = 512;
const int NUM_BLOCKS = 1024;

__global__ void append_kv_kernel(
    const at::Half* k, // [n, nheads, dim]
    const at::Half* v, // [n, nheads, dim]
    at::Half* kv_data, // [?, 2, page_size, nheads, dim]
    const int32_t* indices, // [n]
    int page_size,
    int n,
    int nheads,
    int dim
) {
    int blk_idx = blockIdx.x;
    int thd_idx = threadIdx.x;
    int hdim = nheads * dim;
    int pg = page_size * nheads * dim;
    for (int n_ptr = blk_idx ; n_ptr < n ; n_ptr += NUM_BLOCKS) {
        for (int i = thd_idx ; i < nheads * dim ; i += THREADS_PER_BLOCK) {
            int head_idx = i / dim;
            int page_id = indices[n_ptr / page_size];
            int offset = n_ptr % page_size;
            kv_data[page_id * pg * 2 + offset * hdim + i] = k[n_ptr * hdim + i];
            kv_data[page_id * pg * 2 + pg + offset * hdim + i] = v[n_ptr * hdim + i];
        }
    }
}

void append_kv_launch(
    const at::Half* k, // [n, nheads, dim]
    const at::Half* v, // [n, nheads, dim]
    at::Half* kv_data, // [?, 2, page_size, nheads, dim]
    const int32_t* indices, // [n]
    int page_size,
    int n,
    int nheads,
    int dim 
) {
    append_kv_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        k, v, kv_data, indices, page_size, n, nheads, dim
    );
}