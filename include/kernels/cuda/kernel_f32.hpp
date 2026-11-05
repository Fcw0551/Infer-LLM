#ifdef INFERLLM_WITH_CUDA
#pragma once
#include <cuda_runtime_api.h>
struct Tensor;
namespace CUDA_F32
{

    cudaError_t compute_add(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_mul(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_silu(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_rms_norm(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_get_rows(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_mul_mat(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_reshape(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_rope(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_flash_attn(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_cpy(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_view(Tensor *dst, cudaStream_t stream);
};
#endif // INFERLLM_WITH_CUDA