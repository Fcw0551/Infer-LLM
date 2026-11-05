#ifdef INFERLLM_WITH_CUDA
#include "../../../include/kernels/cuda/kernel_f32.hpp"
namespace CUDA_F32{

    cudaError_t compute_add(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_mul(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_silu(Tensor *dst, cudaStream_t stream);


// ------------------------------------------------------------------------------------------------------------------
    template<int BLOCK_SIZE>
    __global__ void rms_norm(const float *x_base, const float *weight, float *dst_base, const float eps,int64_t hidden_dim) {
        static_assert(BLOCK_SIZE > 0 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0);

        const int64_t row = blockIdx.x;
        const int tid = threadIdx.x;

        const float *x = x_base + row * hidden_dim;
        float *dst = dst_base + row * hidden_dim;

        __shared__ float sums[BLOCK_SIZE];
        __shared__ float inv_rms;

        //  每个线程负责部分元素，局部平方和保存在寄存器中
        float local_sum = 0.0f;     // 不要用dst，直接用一个局部变量，用dst还得读回去显存
        for (int64_t c = tid; c < hidden_dim; c += BLOCK_SIZE){
            const float value = x[c];
            local_sum += value * value;
        }

        //  将各线程的局部结果放进共享内存
        sums[tid] = local_sum;  
        __syncthreads();            // 同步

        //  树形归约：最终 sums[0] 是整个向量的平方和
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2){
            if (tid < stride){
                sums[tid] += sums[tid + stride];        // 交叉分配 合并访存
            }
            __syncthreads(); // 每轮完了之后一定要同步
        }

        //  整个向量共享同一个缩放系数
        if (tid == 0){
            inv_rms = rsqrtf(sums[0] / static_cast<float>(hidden_dim) + eps);
        }
        __syncthreads(); // 一定要加，因为可能tid=0的warp跑的慢，别的warp跑的快，导致下面写回自己的就错了

        //  每个线程写回自己负责的元素
        for (int64_t c = tid; c < hidden_dim; c += BLOCK_SIZE){
            dst[c] = x[c] * inv_rms * weight[c];
        }
    }

    cudaError_t compute_rms_norm(Tensor *dst, cudaStream_t stream){
        // x            : [n_embd , n_tokens]
        // x            : [n_head_embd,head,n_tokens,batch]   //有可能是拆分多头的，但是都是统一沿着第一维度去做rms_norm的
        // weight       : [n_embd,1,1,1]        // gamma γ
        // rms_norm_i   : [n_embd , n_tokens]

        Tensor* src0 = dst->src[0];
        Tensor* src1 = dst->src[1];

        assert(src0->dims[0] == dst->dims[0]);
        assert(src0->dims[1] == dst->dims[1]);
        assert(src0->dims[2] == dst->dims[2]);
        assert(src0->dims[3] == dst->dims[3]);

        INFERLLM_TENSOR_BINARY_LOCALS

        float eps= 0;
        std::memcpy(&eps,&dst->op_params[0],sizeof(eps));

        const float *x_base = static_cast<const float *>(src0->data);
        const float *weight_base = static_cast<const float *>(src1->data);
        float *dst_base = static_cast<float *>(dst->data);
        // 配置核函数
        // 第一维度是要沿着去做rms_norm的，所以有依赖，不同归一化向量之间是没有依赖的
        constexpr int THREADS = 256;
        const int64_t hidden_dim = src0->dims[0];
        const int64_t rows = src0->dims[1] * src0->dims[2] * src0->dims[3];
        rms_norm<THREADS><<<static_cast<unsigned>(rows), THREADS, 0, stream>>>(x_base, weight_base, dst_base, eps, hidden_dim);

        return cudaGetLastError();
    }



// -----------------------------------------------------------------------------------------------------------------------

    __global__ void get_rows(const float *table, const int32_t *indices, float *dst, int64_t hidden_dim, int64_t n_idx){
        // 先找坐标
        const size_t row = blockIdx.x;                              //  计算第几token
        const size_t col = blockIdx.y * blockDim.x + threadIdx.x;
        if (row >= n_idx || col >= hidden_dim){
            return;
        }
        // 计算偏移量
        const size_t token_id = indices[row];
        dst[row*hidden_dim+col] = table[token_id*hidden_dim+col];
    }
    cudaError_t compute_get_rows(Tensor *dst, cudaStream_t stream){
        //  F32
        //  table:   [n_embd, n_vocab, B1, B2]      ← 4 维
        //  indices: [n_idx, B1, B2]                ← 3 维
        //  输出:    [n_embd, n_idx, B1, B2]        ← 4 维
        Tensor *src0 = dst->src[0];
        Tensor *src1 = dst->src[1];
        INFERLLM_TENSOR_BINARY_LOCALS // 宏展开

        assert(src0 && src1);
        assert(src0->dims[0] == dst->dims[0]);
        assert(src1->dims[0] == dst->dims[1]);
        assert(src0->dims[2] == src1->dims[1]);
        assert(src0->dims[3] == src1->dims[2]);

        // 注意tensor的所有材料都应该是在gpu上

        const float *table_base = static_cast<const float *>(src0->data);
        const int32_t *indices_base = static_cast<const int32_t *>(src1->data);
        float *dst_base = static_cast<float *>(dst->data);

        // 配置核函数 todo:
        constexpr int THREADS = 256;
        dim3 block_dims(THREADS,1,1);
        dim3 grid_dims(dims10*dims11*dims12,(dims00+block_dims.x-1)/block_dims.x); // 为了容纳更多的token数量，这里把token放在gridDims.x

        // 启动核函数
        get_rows<<<grid_dims,block_dims,0,stream>>>(table_base,indices_base,dst_base,dims00,dims10*dims11*dims12);  // 不使用共享内存，因为就是简单的gather

        return cudaGetLastError();
    }

// ---------------------------------------------------------------------------------------------------------------
    cudaError_t compute_mul_mat(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_reshape(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_rope(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_flash_attn(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_cpy(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_view(Tensor *dst, cudaStream_t stream);
};
#endif // INFERLLM_WITH_CUDA