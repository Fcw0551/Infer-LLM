#include "../include/scheduler.hpp"

// 每次推理传一张全新的图进来
std::expected<void, std::error_code> Scheduler::execute(Graph& g){
    // 切分子图 
    _subgraphs.clear();
    if (auto r = split_graph(g); !r)
        return r;

    // 分配内存
    if (auto r = alloc_graph(); !r)
        return r;

    // 执行子图
    for (auto &sub : _subgraphs){
        if (auto r = sub.backend->compute(sub.graph); !r)
            return r;
    }

    return {};
}



// 划分子图
std::expected<void, std::error_code> Scheduler::split_graph(Graph &g){
    _subgraphs.clear();
}


// 为计算图中的tensor分配内存
std::expected<void, std::error_code> alloc_graph(graph &g){

}




// -------- helper ---------
// 初始化内存容量
std::unordered_map<Backend*, size_t> Scheduler::init_memory_budgets(Graph &g){
    std::unordered_map<Backend *, size_t> budgets;
    for (Backend *b : _backends){
        budgets[b] = b->get_info().memory_bytes * MEMORY_FACTOR;
    }

    // 扣除权重的占用
     for (Tensor* leaf : g.leafs()) {
        if (leaf->tensorRole == TensorRole::WEIGHT && leaf->backend) {
            budgets[leaf->backend] -= leaf->byte_size();
        }
    }
    return budgets;
}

// 为算子挑选合适的backend
std::expected<Backend*, std::error_code> Scheduler::pick_backend_for_node(Tensor* node, const std::unordered_map<Backend*, size_t>& budgets){
    size_t required = estimate_activation_memory(node);

    // 策略：优先看上游张量在哪个设备，如果它支持且内存够，就跟着上游走
    for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
        Tensor* src = node->src[i];
        if (!src || !src->backend) continue;

        Backend* b = src->backend;
        // 看一下算子是否支持
        if (b->supports_op(node->op, *node) && budgets.at[b] >= required) {
            return b;
        }
    }

    // 这里要么backend不支持这个算子，要么内存容量不够

    // 兜底策略：遍历所有设备，找一个支持且内存够的
    for (Backend* b : _backends) {
        if (b->supports_op(node->op, *node) && budgets.at[b] >= required) {
            return b;
        }
    }
    // 最后如果连cpu都不支持  
    return std::unexpected(std::make_error_code(std::errc::not_enough_memory));
}
size_t Scheduler::estimate_activation_memory(Tensor* node) {
    // 简单版：直接算张量大小
    // （未来如果考虑生命周期复用，这里可做更精细的计算）
    return node->byte_size();
}

