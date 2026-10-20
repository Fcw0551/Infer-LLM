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

    // 五步走，从被权重钉住的算子出发
    // 权重永远不搬，跨设备的只能是激活。
    
    // ----- 先找权重算子，权重在加载的时候已经确定了backend了，根据权重算子确定其消费者算子的backend--------
    // 遍历graph中的node,找其src是权重的，然后确定backend
    // 确定算子是否支持 不支持说明放置和backend能力冲突
    
    // ------- 向前扩展 src->node 按情况进行区分 （核心是避免断层）----------
    // 如果src的所有都是赋值backend且相同，那就看当前backend是否支持当前算子
    // 如果src的backend 指向不同backend 交给第三步
    // 如果src的backend都不同，那就交给第四步

    // ------- 解决（如果src的backend 指向不同backend 交给第三步）--------
    // 统计每个候选设备上的 src 字节数，选最大的（不是数个数）  （这里似乎还有个更好的打分策略，要计算传输设备和传输量之间的关系）
    // 平局优先非 host（GPU）
    // 再平局优先消费者最多的设备   // 这里需要加引用计数在tensor当中，记录后续会被多少的算子用
    // 候选必须 supports_op
    // 一个 host 上的未 pinned 节点，如果它的所有消费者都在同一个非 host 设备 B、B 支持该算子，就把它提升到 B。避免“为了算一个 add 把整段留在 CPU”。

    // ---------解决（如果src的backend都不同，那就交给第四步）---------
    // 从node去反向拉取src，已知消费者，反向拉取生产者

    // tip：234步可以做循环，不断的寻找最优解，一般23步就可以收敛

    //---- 按照连续的backend去切分子图 + 边界拷贝 ------
    // 拷贝结点的生命周期属于谁管？？？
    
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

