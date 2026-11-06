#ifdef INFERLLM_WITH_CUDA
#include "../../../include/kernels/cuda/kernel_f32.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <math_constants.h>
namespace CUDA_F32
{
    // ------------------------------------------------------------------------------------------------------------------

    template <int BLOCK_SIZE>
    __global__ void rms_norm(const float *x_base, const float *weight, float *dst_base, const float eps, int64_t hidden_dim){
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

    __global__ void get_rows(const float *table, const int32_t *indices, float *dst, int64_t hidden_dim, int64_t n_idx)
    {
        // 先找坐标
        const size_t row = blockIdx.x; //  计算第几token
        const size_t col = blockIdx.y * blockDim.x + threadIdx.x;
        if (row >= n_idx || col >= hidden_dim)
        {
            return;
        }
        // 计算偏移量
        const size_t token_id = indices[row];
        dst[row * hidden_dim + col] = table[token_id * hidden_dim + col];
    }
    cudaError_t compute_get_rows(Tensor *dst, cudaStream_t stream)
    {
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
        dim3 block_dims(THREADS, 1, 1);
        dim3 grid_dims(dims10 * dims11 * dims12, (dims00 + block_dims.x - 1) / block_dims.x); // 为了容纳更多的token数量，这里把token放在gridDims.x

        // 启动核函数
        get_rows<<<grid_dims, block_dims, 0, stream>>>(table_base, indices_base, dst_base, dims00, dims10 * dims11 * dims12); // 不使用共享内存，因为就是简单的gather

        return cudaGetLastError();
    }

    // ---------------------------------------------------------------------------------------------------------------

    template <int BLOCK_SIZE>
    __global__ void mul_mat(const float *w_base, const float *x_base, float *dst_base, int64_t hidden_dim)
    {
        static_assert(BLOCK_SIZE > 0 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0);
        const int tid = threadIdx.x;
        const int64_t row = blockIdx.x;
        const int64_t col = blockIdx.y;

        // 当前block负责的位置
        const float *w = w_base + col * hidden_dim;
        const float *x = x_base + row * hidden_dim;
        float *dst = dst_base + col * gridDim.x + row;

        // 共享内存的申请
        __shared__ float sums[BLOCK_SIZE];

        float local_sum = 0.0f;
        for (int64_t i = tid; i < hidden_dim; i += BLOCK_SIZE)
        {
            local_sum + = x[i] * w[i];
        }

        sums[tid] = local_sum; // 放入共享内存，一会做归约
        __syncthreads();       // 同步，确保多有的warp都放到共享内存当中

        // 归约（交叉分配）
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
        {
            if (tid < stride)
            {
                sums[tid] += sums[stride + tid];
            }
            __syncthreads(); // 每轮都要同步才能进入下一轮
        }
        // 共享内存第一个位置就是放着归约的结果了
        if (tid == 0)
            dst[0] = sums[0];
    }
    cudaError_t compute_mul_mat(Tensor *dst, cudaStream_t stream)
    {
        // w     : [n_embd, n_out]
        // x     : [n_embd, n_tokens]
        // out   : [n_out, n_tokens]
        //  x*w（转置）
        Tensor *src0 = dst->src[0];
        Tensor *src1 = dst->src[1];
        INFERLLM_TENSOR_BINARY_LOCALS
        assert(dims00 == dims10);
        assert(dims01 == dims0);
        assert(dims11 == dims1);

        const float *w_base = static_cast<const float *>(src0->data);
        const float *x_base = static_cast<const float *>(src1->data);
        float *dst_base = static_cast<float *>(dst->data);

        // 矩阵相乘就是对应位置相乘再相加
        // 一：可以利用局部变量先把中间值存下来，避免频繁的读取显存
        // 二：利用共享内存归约
        constexpr int THREADS = 256; // 一个block的大小
        // grid的大小就是dst的大小
        dim3 grid(static_cast<unsigned>(dims0), static_cast<unsigned>(dims1));
        mul_mat<THREADS><<<(grid, dims1, 1), THREADS, 0, stream>>>(w_base, x_base, dst_base, dims00);
        return cudaGetLastError();
    }

    // ------------------------------------------------------------------------------------------------------------------

    // 先暂时实现拷贝版本，因为view还没有完整的做好
    cudaError_t compute_reshape(Tensor *dst, cudaStream_t stream)
    {
        if (!dst || !dst->src[0])
        {
            return cudaErrorInvalidValue;
        }
        Tensor *src0 = dst->src[0];
        if (src0->dataType != dst->dataType || src0->element_count() != dst->element_count())
        {
            return cudaErrorInvalidValue;
        }

        // 与 CPU 版本相同：前提是输入、输出均为连续布局
        // 形状和 strides 已在建图时设置，这里只复制数据
        const size_t bytes = src0->byte_size();
        if (bytes == 0)
        {
            return cudaSuccess;
        }
        if (!src0->data || !dst->data)
        {
            return cudaErrorInvalidValue;
        }
        if (src0->data == dst->data)
        {
            return cudaSuccess;
        }

        // 在调用方的流上排队，不在算子内同步；直接返回复制 API 的错误码
        return cudaMemcpyAsync(dst->data, src0->data, bytes, cudaMemcpyDeviceToDevice, stream);
    }

    // 一个 block 处理一个 (token, head)，每个线程旋转前后半区的一对元素
    // x/dst: [hidden_dim, n_head, n_token, 1]，theta: [hidden_dim/2, n_ctx]
    __global__ void rope(const float *x_base, const int32_t *inp_pos_base, const float *theta_base, float *dst_base, int64_t hidden_dim, int64_t n_head, int64_t n_ctx)
    {
        const int64_t row = blockIdx.x;
        const int64_t token = row / n_head;           // 计算的是第几个token
        const int64_t position = inp_pos_base[token]; // 该token的实际的下标
        // 位置值在显存中检查；非法位置触发设备错误，由调用方同步时检查
        if (position < 0 || position >= n_ctx)
        {
            __trap();
            return;
        }
        // 从哪里取，放到哪里
        const int64_t n_pairs = hidden_dim / 2;
        const float *x = x_base + row * hidden_dim;
        float *out = dst_base + row * hidden_dim;
        const float *theta = theta_base + position * n_pairs;

        for (int64_t j = threadIdx.x; j < n_pairs; j += blockDim.x)
        {
            float s, c;
            sincosf(theta[j], &s, &c); // 计算sinθ和cosθ
            // 先读完一对输入，允许 dst 与 x 完全同址
            const float x0 = x[j];
            const float x1 = x[j + n_pairs];
            out[j] = x0 * c - x1 * s;
            out[j + n_pairs] = x0 * s + x1 * c;
        }
    }

    cudaError_t compute_rope(Tensor *dst, cudaStream_t stream)
    {
        if (!dst || !dst->src[0] || !dst->src[1] || !dst->src[2])
        {
            return cudaErrorInvalidValue;
        }
        const Tensor *src0 = dst->src[0]; // x
        const Tensor *src1 = dst->src[1]; // inp_pos
        const Tensor *src2 = dst->src[2]; // theta
        const int64_t hidden_dim = src0->dims[0];
        const int64_t n_head = src0->dims[1];
        const int64_t n_token = src0->dims[2];
        const int64_t n_pairs = hidden_dim / 2;

        if (n_token == 0)
        {
            return cudaSuccess;
        }

        const float *x_base = static_cast<const float *>(src0->data);
        const int32_t *inp_pos_base = static_cast<const int32_t *>(src1->data);
        const float *theta_base = static_cast<const float *>(src2->data);
        float *dst_base = static_cast<float *>(dst->data);
        constexpr unsigned THREADS = 64;
        const dim3 block(THREADS);
        const dim3 grid(static_cast<unsigned>(n_head * n_token));
        rope<<<grid, block, 0, stream>>>(x_base, inp_pos_base, theta_base, dst_base, hidden_dim, n_head, src2->dims[1]);
        return cudaGetLastError();
    }

    // -----------------------------------------------------------------------------------------

    cudaError_t compute_cpy(Tensor *dst, cudaStream_t stream)
    {
        if (!dst || !dst->src[0])
        {
            return cudaErrorInvalidValue;
        }
        const Tensor *src = dst->src[0];

        // ----- 检查源和目标 ------
        if (!src->data || !dst->data || src->dataType != dst->dataType || src->dims != dst->dims)
        {
            return cudaErrorInvalidValue;
        }
        cudaMemcpyAsync(dst->data, src->data, src->byte_size(), cudaMemcpyDeviceToDevice, stream);
    }

    // -----------------------------------------------------------------------------------------

    cudaError_t compute_view(Tensor *dst, cudaStream_t stream)
    {
        return cudaSuccess;
    }

    // ------------------------------------------------------------------------------------

    template <int BLOCK_SIZE>
    __global__ void score(const float *q_base, const float *k_base, float *scores, int64_t n_tokens_q, int64_t n_tokens_kv, int64_t n_heads, int64_t n_heads_kv,int64_t head_dim,int64_t group_size, int64_t n_past, float scale){
        static_assert(BLOCK_SIZE > 0 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0);
        // 使用共享内存 + 局部变量（防止重复的读取dst）
        int tid = threadIdx.x;
        int64_t row = blockIdx.x; // 第几个任务

        // 以下的s标识score矩阵
        int64_t s_h = row / (n_tokens_q * n_tokens_kv);                             // 相当于要第几个头
        int64_t s_y = (row - (s_h * n_tokens_q * n_tokens_kv)) / n_tokens_kv;       // 相当于要q的第几个token
        int64_t s_x = row - (s_h * n_tokens_q * n_tokens_kv) - (s_y * n_tokens_kv); // 相当于要k的第几个token

        // GQA?
        const int64_t kv_head = s_h / group_size;

        // 同一个 block 的判断一致，可以在归约同步前整体返回
        if (s_x > n_past + s_y){
            if (tid == 0){
                scores[row] = -CUDART_INF_F;
            }
            return;
        }
        // 不会被掩码干扰的位置，score当中应该计算的位置

        // 计算src的位置
        const float *q_start = q_base + s_y * head_dim * n_heads + s_h * head_dim;
        const float *k_start = k_base + s_x * head_dim * n_heads_kv + kv_head * head_dim;

        __shared__ float sums[BLOCK_SIZE]; // 共享内存

        float local_sum = 0.0f;
        
        for (int64_t i = tid; i < head_dim; i += BLOCK_SIZE){
            local_sum += q_start[i] * k_start[i];
        }
        sums[tid] = local_sum;
        __syncthreads();

        // 开始归约
        for (int strides = BLOCK_SIZE / 2; strides > 0; strides /= 2){
            if(tid<strides){
                sums[tid]+=sums[tid+strides];
            }
            __syncthreads();
        }
        // 共享内存第一个就是放着每个block的每个任务的结果
        if (tid == 0){
            scores[row] = sums[0] * scale;
        }
    }
    template<int BLOCK_SIZE>
    __global__ void softmax(float* scores_base, float * dst_base,const float* v_base,int64_t n_heads,int64_t n_heads_kv,int64_t head_dim,int64_t n_tokens_q,int64_t n_tokens_kv,int64_t group_size){
        // 一个block处理一行
        int tid = threadIdx.x;
        int64_t row = blockIdx.x;
        float *scores = scores_base + row * n_tokens_kv;

        // 共享内存进行归约一行
       __shared__ float scratch[BLOCK_SIZE];
       float local_max = -CUDART_INF_F;
       for (int64_t i = tid; i < n_tokens_kv; i += BLOCK_SIZE){
           local_max = fmaxf(local_max, scores[i]);
       }
       scratch[tid] = local_max;
       __syncthreads();
       for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1){
           if (tid < stride){
               scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
           }
           __syncthreads();
       }
       const float row_max = scratch[0];    // 一行里面的最大值
       __syncthreads();


       float local_sum = 0.0f;
       for (int64_t i = tid; i < n_tokens_kv; i += BLOCK_SIZE){
           const float value = expf(scores[i] - row_max);
           scores[i] = value; // 原地保存指数值，后面归一化时使用
           local_sum += value;
       }
       scratch[tid] = local_sum;
       __syncthreads();

       // 再次归约，此时求的是分母
       for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1){
           if (tid < stride){
               scratch[tid] += scratch[tid+stride];
           }
           __syncthreads();
       }
       const float sum = scratch[0];
       __syncthreads();

       // 每个数值直接除以分母即可
       for (int64_t i = tid; i < n_tokens_kv; i += BLOCK_SIZE){
           scores[i]/=sum;
       }
       __syncthreads();

       // Sh * Vh
       const int64_t q_head = row / n_tokens_q;
       const int64_t q_token = row % n_tokens_q;
       const int64_t kv_head = q_head / group_size;

       // 输出布局：[head_dim, n_heads, n_tokens_q]
       float *out = dst_base + (q_token * n_heads + q_head) * head_dim;

       // V 布局：[head_dim, n_heads_kv, n_tokens_kv]
       for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE){
           float acc = 0.0f;

           for (int64_t k = 0; k < n_tokens_kv; ++k){
               const int64_t offset =(k * n_heads_kv + kv_head) * head_dim + d;

               acc += scores[k] * v_base[offset];
           }

           out[d] = acc;
       }
    }

    cudaError_t compute_flash_attn(Tensor *dst, cudaStream_t stream){
        // q[head_dim, n_heads,n_tokens_q]
        // k[head_dim, n_heads_kv,n_tokens_kv]
        // v[head_dim, n_heads_kv,n_tokens_kv]
        // dst[head_dim, n_heads,n_tokens_q]

        if (!dst || !dst->src[0] || !dst->src[1]) return cudaErrorInvalidValue;
        Tensor *src0 = dst->src[0];
        Tensor *src1 = dst->src[1];
        Tensor *src2 = dst->src[2];

        const float *q_base = static_cast<const float *>(src0->data);
        const float *k_base = static_cast<const float *>(src1->data);
        const float *v_base = static_cast<const float *>(src2->data);


        const int64_t n_tokens_q = src0->dims[2];
        const int64_t n_tokens_kv = src1->dims[2];
        const int64_t n_heads = src0->dims[1];
        const int64_t head_dim = src0->dims[0];
        const int64_t n_heads_kv = src1->dims[1];
        const int64_t group_size = n_heads / n_heads_kv;
        
        // Tensor 描述在 CPU，只有 data 指向显存；op_params 可以直接 memcpy
        AttnParams p{};
        std::memcpy(&p, dst->op_params, sizeof(p));

        const float scale = p.scale > 0.0f ? p.scale : 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int64_t score_count = n_tokens_q * n_tokens_kv * n_heads;
        float *scores = nullptr;
        cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&scores),static_cast<size_t>(score_count) * sizeof(float));
        if (error != cudaSuccess){
            return error;
        }

        constexpr unsigned THREADS = 32;
        dim3 block(THREADS);
        dim3 grid(static_cast<unsigned>(score_count));
        score<THREADS><<<grid, block, 0, stream>>>(q_base, k_base, scores,n_tokens_q, n_tokens_kv, n_heads, n_heads_kv, head_dim, group_size,static_cast<int64_t>(p.n_ctx), scale);
        cudaError_t err = cudaGetLastError();
        // 一行一个softmax  所以开的个数就等于头数*行数即可
        if (err == cudaSuccess){
            softmax<THREADS><<<n_heads * n_tokens_q, block, 0, stream>>>(scores, dst_base, v_base, n_heads, n_heads_kv, head_dim, n_tokens_q, n_tokens_kv, group_size);
            err = cudaGetLastError();
        }

        // 等待已提交的任务结束，再清理 scores
        // 即使第二次启动失败，也要考虑第一次已经提交的任务
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        cudaError_t free_err = cudaFree(scores);

        // 优先保留最先发现的错误
        if (err != cudaSuccess)
            return err;
        if (sync_err != cudaSuccess)
            return sync_err;
        return free_err;
    }

// ---------------------------------------------------------------------------------------

    __global__ void add(const float *a_base, const float *b_base, float *dst_base,int len){
        const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x * 2 + threadIdx.x;

        // 第一个元素
        if (offset < len){
            dst_base[offset] = a_base[offset] + b_base[offset];
        }

        // 第二个元素
        const int64_t second = offset + blockDim.x;
        if (second < len){
            dst_base[second] = a_base[second] + b_base[second];
        }
    }
    cudaError_t compute_add(Tensor *dst, cudaStream_t stream){
        Tensor* src0 = dst->src[0];
        Tensor* src1 = dst->src[1];

        INFERLLM_TENSOR_BINARY_LOCALS

        // a[dims,token]
        const float *a_base = static_cast<const float *>(src0->data);
        const float *b_base = static_cast<const float *>(src1->data);
        float *dst_base = static_cast<float *>(dst->data);

        const int64_t len = src0->element_count();
        if (len == 0){
            return cudaSuccess;
        }
        // 配置核函数
        constexpr unsigned THREADS = 32;
        constexpr unsigned ELEMENTS_PER_BLOCK = THREADS * 2;

        dim3 block(THREADS);
        dim3 grid(static_cast<unsigned>((len - 1) / ELEMENTS_PER_BLOCK + 1));

        add<<<grid, block, 0, stream>>>(a_base, b_base, dst_base, len);

        return cudaGetLastError();
    }

    cudaError_t compute_mul(Tensor *dst, cudaStream_t stream);

    cudaError_t compute_silu(Tensor *dst, cudaStream_t stream);
};
#endif // INFERLLM_WITH_CUDA
