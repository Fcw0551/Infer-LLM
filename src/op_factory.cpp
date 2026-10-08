#pragma once
#include "tensor.hpp"
#include "graph.hpp"
#include <initializer_list>
#include <string>

// 注意力算子的参数包
struct AttnParams {
    uint32_t n_head    = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_ctx     = 0;
    float    scale     = 0.0f;   // 通常 = 1/sqrt(head_dim)
    bool     causal    = true;
};

class OpFactory {
public:
    // 基础算子   
    static Tensor* add(Graph& g, Tensor* a, Tensor* b);
    static Tensor* sub(Graph& g, Tensor* a, Tensor* b);
    static Tensor* mul(Graph& g, Tensor* a, Tensor* b);
    static Tensor* div(Graph& g, Tensor* a, Tensor* b);

    // 矩阵
    // w: [n_in, n_out], x: [n_in, n_tokens] → out: [n_out, n_tokens]
    static Tensor* mul_mat(Graph& g, Tensor* w, Tensor* x);

    // 归一化 
    static Tensor* rms_norm(Graph& g, Tensor* x, Tensor* weight, float eps);
    static Tensor* layer_norm(Graph& g, Tensor* x, Tensor* weight, Tensor* bias, float eps);

    // 激活
    static Tensor* silu(Graph& g, Tensor* x);
    static Tensor* gelu(Graph& g, Tensor* x);

    // 位置编码 
    // freqs: 预计算的 sin/cos 表，作为 src[2]
    static Tensor* rope(Graph& g, Tensor* x, Tensor* pos, Tensor* freqs, float theta);

    // 注意力
    static Tensor* soft_max(Graph& g, Tensor* x, Tensor* mask, float scale);
    static Tensor* flash_attn(Graph& g, Tensor* q, Tensor* k, Tensor* v,
                              const AttnParams& p);

    // 索引
    // table: [n_embd, n_vocab], indices: [n_tokens] → out: [n_embd, n_tokens]
    static Tensor* get_rows(Graph& g, Tensor* table, Tensor* indices);

    // 形状
    static Tensor* reshape(Graph& g, Tensor* x, std::initializer_list<int64_t> shape);
    static Tensor* permute(Graph& g, Tensor* x, std::initializer_list<int> axes);
    static Tensor* cont(Graph& g, Tensor* x);    // 强制连续
    static Tensor* view(Graph& g, Tensor* x, std::initializer_list<int64_t> shape);

private:
    // 唯一创建中间张量的地方
    static Tensor* make(Graph& g, DType type,
                        std::initializer_list<int64_t> shape,
                        std::string_view name);
};

