#pragma once
#include <unordered_set>
#include <memory>
#include <system_error>
#include <functional>
#include <cmath>
#include <iostream>
#include "tensor.hpp"
#include "util.hpp"
class Graph {
public:
    // 构建 API 
    Tensor* add_node(Tensor* t);
    Tensor* add_node(std::unique_ptr<Tensor> t);
    void mark_input(Tensor* t){
        _inputs.push_back(t);
    }
    void mark_output(Tensor* t){
        _outputs.push_back(t);
    }

    // 执行
    std::expected<void, std::error_code> compute();
    std::expected<void, std::error_code> compute(Device& backend);

    // 访问
    const std::vector<Tensor*>& inputs()  const { return _inputs; }
    const std::vector<Tensor*>& outputs() const { return _outputs; }
    const std::vector<Tensor*>& nodes()   const { return _nodes; }   // 拓扑序
    const std::vector<Tensor*>& leafs()   const { return _leafs; }

private:
    void build_topo_order(Tensor* t);

    //标记图有没有发生变化
    bool _topo = false;

    // 所有权
    std::vector<std::unique_ptr<Tensor>> _owned;   // 所有中间张量的所有权

    //  边界
    std::vector<Tensor*> _inputs;                  // build_graph标记的输入
    std::vector<Tensor*> _outputs;                 // build_graph标记的输出

    // 拓扑排序结果
    std::vector<Tensor*> _nodes;                   // 按拓扑序排列的内部节点
    std::vector<Tensor*> _leafs;                   // 所有叶子节点（权重、输入）

    // 拓扑排序的临时状态
    std::unordered_set<Tensor*> _visited;          // DFS 去重

};


//承担动态资源的集合
class GraphContext{
public:
    GraphContext() = default;

    // 禁止拷贝（内部持有张量）
    GraphContext(const GraphContext &) = delete;
    GraphContext &operator=(const GraphContext &) = delete;
    GraphContext(GraphContext &&) noexcept = default;
    GraphContext &operator=(GraphContext &&) noexcept = default;

    // 输入
    void set_input_tokens(Tensor *t) { _inp_tokens = t; }
    void set_input_pos(Tensor *t) { _inp_pos = t; }

    Tensor *get_input_tokens() const { return _inp_tokens; }
    Tensor *get_input_pos() const { return _inp_pos; }

    // RoPE 频率表缓存
    // 返回 [n_ctx, head_dim/2] 的 θ 表，每行是 pos * freq_i
    std::expected<Tensor *, std::error_code> get_rope_theta_table(float theta_base,uint32_t head_dim,uint32_t n_ctx);

    //  张量所有权池
    // GraphContext 内部创建的张量（输入张量、freqs）都挂在这
    Tensor *make_tensor(std::string name, DataType type, std::initializer_list<int64_t> shape);

private:
    // 输入
    Tensor *_inp_tokens = nullptr; // [n_tokens]  I32
    Tensor *_inp_pos = nullptr;    // [n_tokens]  I32

    // theata表缓存
    struct RopeKey
    {
        float theta_base;
        uint32_t head_dim;
        uint32_t n_ctx;
        bool operator==(const RopeKey &o) const noexcept
        {
            return theta_base == o.theta_base && head_dim == o.head_dim && n_ctx == o.n_ctx;
        }
    };
    //hash函数
    struct RopeKeyHash
    {
        size_t operator()(const RopeKey &k) const noexcept
        {
            size_t h = std::hash<float>{}(k.theta_base);
            h ^= std::hash<uint32_t>{}(k.head_dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>{}(k.n_ctx) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    std::unordered_map<RopeKey, Tensor* , RopeKeyHash> _rope_theta_cache;

    // 所有权池
    // 所有 GraphContext 创建的张量放这里，析构时统一释放
    std::vector<std::unique_ptr<Tensor>> _owned;
};