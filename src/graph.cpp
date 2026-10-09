#pragma once
#include "../include/graph.hpp"
//------Graph--------

// 构建 API
Tensor *Graph::add_node(Tensor *t){
    Tensor* raw = t;
    _owned.push_back(std::unique_ptr<Tensor>(t));   
    return raw;
}
Tensor *Graph::add_node(std::unique_ptr<Tensor> t){
    Tensor *raw = t.get();
    _owned.push_back(std::move(t)); // 接管所有权
    return raw;
}


// 执行
std::expected<void, std::error_code> compute(Device &backend);





//承担动态资源的集合
//-------class GraphContext---------

// RoPE 频率表缓存
// 返回 [n_ctx, head_dim/2] 的 θ 表，每行是 pos * freq_i
std::expected<Tensor *, std::error_code> GraphContext::get_rope_theta_table(float theta_base, uint32_t head_dim, uint32_t n_ctx){
    RopeKey key{theta_base, head_dim, n_ctx};
    auto it = _rope_theta_cache.find(key);
    if (it != _rope_theta_cache.end())
        return it->second;

    const uint32_t half = head_dim / 2;

    // 创建张量[half, n_ctx]
    Tensor *t = make_tensor("rope_theta",DataType::GGML_TYPE_F32, {half, n_ctx, 1, 1});

    // 分配 data 纯数据块
    t->data = std::malloc(half * n_ctx * sizeof(float));
    if (t->data == nullptr){
        return std::unexpected(std::make_error_code(std::errc::not_enough_memory));
    }
    float *data = static_cast<float *>(t->data);

    // 先算 freq_i（只有 half 个不同的值，两两一组，多头共用）
    std::vector<double> freq(half);
    for (uint32_t i = 0; i < half; ++i){
        freq[i] = 1.0f / std::pow(theta_base, 2.0f * i / head_dim);
    }

    // 填充 θ[pos][i] = pos * freq_i  张量[half, n_ctx]
    for (uint32_t pos = 0; pos < n_ctx; ++pos){
        float *row = data + pos * half; // 先计算是哪行，要跳过之前的，整个的内存是一维线性的
        for (uint32_t i = 0; i < half; ++i)
        {
            row[i] = static_cast<float>(static_cast<double>(pos) * freq[i]);
        }
    }

    _rope_theta_cache[key] = t; // 缓存起来
    return t;
}

// 张量所有权池
// GraphContext 内部创建的张量（输入张量、rope_theta表）都挂在这
Tensor *GraphContext::make_tensor(std::string name, DataType type, std::initializer_list<int64_t> shape){
    std::unique_ptr<Tensor> p = std::make_unique<Tensor>(name, type, shape);
    Tensor* raw=p.get();
    _owned.push_back(std::move(p));  
    return raw;
}

