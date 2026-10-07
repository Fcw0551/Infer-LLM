#pragma once 
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