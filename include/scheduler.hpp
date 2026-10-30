#pragma once 
#include <vector>
#include <memory>
#include <utility>
#include<unordered_map>
#include"graph.hpp"
#include "util.hpp"
#include "../include/backend.hpp"  
#include "../include/activationManager.hpp" 
//#include "kvcacheManager.hpp"

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
    Graph graph;                   // 引用模式图（节点只存指针；边界拷贝由总图持有）
    std::vector<Tensor *> inputs;  // 从其他设备传进来的边界张量（拷贝源）
    std::vector<Tensor *> outputs; // 要传给其他设备的边界张量
    // 本设备上的边界拷贝 {src, dst}：执行本子图之前依次做掉
    // dst 通过总图 g 的 adopt(unique_ptr) 创建，生命周期跟总图走，不随子图释放
    std::vector<std::pair<Tensor *, Tensor *>> copies;
};



class Scheduler{
public:
    Scheduler(std::vector<Backend* >& backend)
    :_backends(backend)
    ,_ac(backend)
    {}
    // 服务scheduler
    void add_backend(Backend& backend){
        _backends.push_back(&backend);
    }
    // TODO : 现在暂时没有绑定graph，因为目前每次都要decode
    // 执行
    std::expected<void, std::error_code> execute(Graph &g);

private:

    // 划分子图
    std::expected<void, std::error_code> split_graph(Graph &g);
    // 为计算图中的tensor分配内存
    std::expected<void, std::error_code> alloc_graph(Graph &g);
    // 按顺序执行，传输同步
    std::expected<void, std::error_code> compute_graph(Graph &g);
    // 跨设备拷贝：先试 dst 设备的 P2P 高速路径，不行再退回 host 中转
    std::expected<void, std::error_code> copy_tensor(Tensor *src, Tensor *dst);

    // ------helper-------
    // 初始化内存容量
    std::expected<std::unordered_map<Backend *, size_t>, std::error_code> init_memory_budgets(Graph &g);
    // 制作一张规划计算图（后续传给activationManager）
    std::expected<void,std::error_code> build_activation_graph(Graph &graph);
private:
    //Graph* _graph;                                                            // TODO 现在暂时没有绑定graph，因为目前每次都要decode
    std::vector<Backend *> _backends;                                           // 管理可用计算设备
    std::vector<Subgraph> _subgraphs;                                           // 子图
    Graph _activation_graph;                                                    // 给_ac的规划图
    ActivationManager _ac;                                                      // activation管理类
    //KVCacheManager* _kv = nullptr;                                            // kv管理类
};
