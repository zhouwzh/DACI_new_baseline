#include <torch/extension.h>

#include <vector>


void append_kv_launch(
    const at::Half* k, // [n, nheads, dim]
    const at::Half* v, // [n, nheads, dim]
    at::Half* kv_data, // [?, 2, page_size, nheads, dim]
    const int32_t* indices, // [n]
    int page_size,
    int n,
    int nheads,
    int dim 
);

void append_page_kv(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor kv_indices,
    torch::Tensor kvdata) {
        TORCH_CHECK(k.scalar_type() == torch::kFloat16, "k must be float16");
        TORCH_CHECK(v.scalar_type() == torch::kFloat16, "v must be float16");
        TORCH_CHECK(kvdata.scalar_type() == torch::kFloat16, "kvdata must be float16");
        TORCH_CHECK(kv_indices.scalar_type() == torch::kInt32, "indices must be int32");

        // K [seqlen, num_heads, head_dim]
        TORCH_CHECK_EQ(k.dim(), 3);
        TORCH_CHECK_EQ(v.dim(), 3);
        TORCH_CHECK_EQ(k.size(0), v.size(0));
        TORCH_CHECK_EQ(k.size(1), v.size(1));
        TORCH_CHECK_EQ(k.size(2), v.size(2));

        TORCH_CHECK_EQ(kv_indices.dim(), 1);
        return append_kv_launch(
            k.data_ptr<at::Half>(),
            v.data_ptr<at::Half>(),
            kvdata.data_ptr<at::Half>(),
            kv_indices.data_ptr<int32_t>(),
            kvdata.size(2),
            k.size(0),
            k.size(1),
            k.size(2)
        );
    }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("append_page_kv", &append_page_kv, "append kv to paged cache");
}