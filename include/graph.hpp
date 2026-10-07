#pragma once
#include <unordered_map>
class Graph {
public:
    // 构建 API 
    Tensor* add_node(Tensor* t);
    void mark_input(Tensor* t);
    void mark_output(Tensor* t);

    // 执行
    std::expected<void, std::error_code> compute(Backend& backend);

    // 访问
    const std::vector<Tensor*>& inputs()  const { return _inputs; }
    const std::vector<Tensor*>& outputs() const { return _outputs; }
    const std::vector<Tensor*>& nodes()   const { return _nodes; }   // 拓扑序
    const std::vector<Tensor*>& leafs()   const { return _leafs; }

private:
    void build_topo_order(Tensor* t);

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

    //  RoPE 频率表缓存
    // 同一个 (theta, n_dims, n_ctx) 只算一次，所有层、多次前向复用
    Tensor *get_rope_freqs(float theta, uint32_t n_dims, uint32_t n_ctx);

    //  张量所有权池
    // GraphContext 内部创建的张量（输入张量、freqs）都挂在这
    Tensor *make_tensor(DType type, std::initializer_list<int64_t> shape,
                        std::string name);

private:
    // 输入
    Tensor *_inp_tokens = nullptr; // [n_tokens]  I32
    Tensor *_inp_pos = nullptr;    // [n_tokens]  I32

    // 缓存
    struct RopeKey
    {
        float theta;
        uint32_t n_dims;
        uint32_t n_ctx;
        bool operator==(const RopeKey &o) const
        {
            return theta == o.theta && n_dims == o.n_dims && n_ctx == o.n_ctx;
        }
    };
    struct RopeKeyHash
    {
        size_t operator()(const RopeKey &k) const
        {
            return std::hash<float>{}(k.theta) ^ (std::hash<uint32_t>{}(k.n_dims) << 1) ^ (std::hash<uint32_t>{}(k.n_ctx) << 2);
        }
    };
    std::unordered_map<RopeKey, Tensor *, RopeKeyHash> _rope_cache;

    // 所有权池
    // 所有 GraphContext 创建的张量放这里，析构时统一释放
    std::vector<std::unique_ptr<Tensor>> _owned;
};