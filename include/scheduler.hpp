#pragma once 
#include <vector>
#include <memory>
#include<unordered_map>
#include"graph.hpp"
#include "util.hpp"

// 承担计算资源与计算任务的调度
    // 管理可用计算设备
    // 图切分
    // 内存分配
    // 跨设备传输
    // 执行与同步

// 内存预算
#define MEMORY_FACTOR 0.8
struct MemoryBudget {
    size_t weight_bytes = 0;       // 权重的常驻内存
    size_t activation_bytes = 0;   // 激活值的临时内存 
    size_t kv_cache_bytes = 0;     // KV Cache 的预留
    
    size_t total_required() const {
        return weight_bytes + activation_bytes + kv_cache_bytes;
    }
};


// 子图
struct Subgraph{
    Backend *backend = nullptr;    // 这个子图归谁管
    Graph graph;                   // 引用模式图（只存 node 指针，不管生命周期）
    std::vector<Tensor *> inputs;  // 从其他设备传进来的边界张量
    std::vector<Tensor *> outputs; // 要传给其他设备的边界张量
};



class Scheduler{
public:
    Scheduler(std::vector<Backend* >& backend)
    , _backends(backend)
    {}
    // 服务scheduler
    void add_backend(Backend& backend){
        _backends.push_back(*backend);
    }
    // TODO : 现在暂时没有绑定graph，因为目前每次都要decode
    // 执行
    std::expected<void, std::error_code> execute(graph &g);

private:

    // 划分子图
    std::expected<void, std::error_code> split_graph(graph &g);
    // 为计算图中的tensor分配内存
    std::expected<void, std::error_code> alloc_graph(graph &g);
    // 按顺序执行，传输同步
    std::expected<void, std::error_code> compute_graph(graph &g);

    // helper
    // 初始化内存容量
    std::unordered_map<Backend *, size_t> Scheduler::init_memory_budgets(Graph &g);
    // 为算子挑选合适的backend
    std::expected<Backend *, std::error_code> Scheduler::pick_backend_for_node(Tensor *node, const std::unordered_map<Backend *, size_t> &budgets);
    // 计算tensor的大小 TODO
    size_t Scheduler::estimate_activation_memory(Tensor *node);

private:
    //Graph* _graph;                                      // TODO 现在暂时没有绑定graph，因为目前每次都要decode
    std::vector<Backend *> _backends;                     // 管理可用计算设备
    std::vector<Subgraph> _subgraphs;                     // 子图
};
