#pragma once
#include <cstdint>

// 注意力算子的参数包
struct AttnParams {
    // 结构
    uint32_t n_head    = 0;      // Q 头数
    uint32_t n_head_kv = 0;      // KV 头数 (GQA)
    uint32_t head_dim  = 0;      // 单头维度 (Qwen3=128)

    // 数学
    float    scale     = 0.0f;   // 1/sqrt(head_dim)

    // 掩码与位置
    bool     causal    = true;   // 是否因果
    uint32_t n_ctx     = 0;      // 历史上下文长度 (用于 KV Cache)
    uint32_t sliding_window = 0; // 0 表示禁用
    
    // 预留对齐
    uint32_t _pad[2] = {0, 0};
};
