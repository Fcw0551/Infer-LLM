#include "../include/graph.hpp"
#include "../include/backend.hpp"
#include "../include/kvcache/kvcacheManager.hpp"
#include <new>
//------Graph--------

// 构建 API
Tensor *Graph::add_node(Tensor *t){
    Tensor* raw = t;
    _owned.push_back(std::unique_ptr<Tensor>(t));
    _topo=true;   
    return raw;
}
Tensor *Graph::add_node(std::unique_ptr<Tensor> t){
    Tensor *raw = t.get();
    _owned.push_back(std::move(t)); // 接管所有权
    _topo=true;
    return raw;
}


// 执行 建图
std::expected<void, std::error_code> Graph::compute(){
    if (_outputs.empty()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    // 图变了，重建拓扑序
    if (_topo){
        _visited.clear();
        _nodes.clear();
        _leafs.clear();

        for (Tensor *out : _outputs){
            //从输出结点开始建立拓扑图
            build_topo_order(out);
        }
        _topo = false;

        std::cout << "topo: leafs=" << _leafs.size() << ", nodes=" << _nodes.size() << "\n";
    }

    //  ---图没变，不需要重建拓扑---
    
    // 叶子节点：检查数据是否就绪
    for (Tensor *leaf : _leafs){
        if (!leaf->data){
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        }
    }

    // // 按拓扑序执行内部节点 （之前调试剩下的）
    // for (Tensor *node : _nodes){
    //     // if (auto r = backend.compute(*node); !r){
    //     //     return r;
    //     // }
    //    // std::cout<<"正在执行算子："<<node->name<<std::endl;
    // }

    return {};
}


void Graph::build_topo_order(Tensor* t){
    if (!t) return;
    if (!_visited.insert(t).second) return; // 已访问过

    // 后序遍历：先处理所有依赖，再处理自己
    for(auto &e: t->src){
        if(!e) break;
        build_topo_order(e);
    }

    if (t->op == OperationType::GGML_OP_NONE){
        _leafs.push_back(t); // 叶子：权重、输入
    }
    else{
        _nodes.push_back(t); // 内部节点：按拓扑序排列
    }
}






//承担动态资源的集合
//-------class GraphContext---------

// RoPE 频率表缓存
// 返回 [n_ctx, head_dim/2] 的 θ 表，每行是 pos * freq_i
std::expected<Tensor *, std::error_code> GraphContext::get_rope_theta_table(float theta_base, uint32_t head_dim, uint32_t n_ctx){
    if (!_cpu_backend || !_cpu_backend->is_host()) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    RopeKey key{theta_base, head_dim, n_ctx};
    auto it = _rope_theta_cache.find(key);
    if (it != _rope_theta_cache.end())
        return it->second;

    const uint32_t half = head_dim / 2;

    // 创建张量[half, n_ctx]
    Tensor *t = make_tensor("rope_theta",DataType::GGML_TYPE_F32, {half, n_ctx, 1, 1});
    // 数据由 CPU 的 new[] 分配，标记真实来源以便调度器插入跨设备复制
    t->backend = _cpu_backend;
    t->device = DeviceType::CPU;

    // 分配 data 纯数据块
    auto buffer = std::unique_ptr<float[]>(new (std::nothrow) float[static_cast<size_t>(half) * n_ctx]);
    if (!buffer){
        return std::unexpected(std::make_error_code(std::errc::not_enough_memory));
    }
    t->data = buffer.get();
    _rope_buffers.push_back(std::move(buffer));
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

std::expected<void, std::error_code> GraphContext::set_kv_cache(KVCacheManager *kv, SequenceId id) noexcept{
    if (!_kv_accesses.empty() && (kv != _kv_manager || id != _sequence_id)) {
        return std::unexpected(std::make_error_code(std::errc::device_or_resource_busy));
    }
    _kv_manager = kv;
    _sequence_id = id;
    return {};
}

std::expected<void, std::error_code> GraphContext::release_kv_access(){
    // 没有访问需要归还，重复调用也成功
    if (_kv_accesses.empty()){
        return {};
    }

    if (!_kv_manager){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    for (auto &access : _kv_accesses){
        auto result = _kv_manager->release_layer(access);
        if (!result){
            return std::unexpected(result.error());
        }
    }

    // 全部归还后再销毁这些访问对象
    _kv_accesses.clear();
    return {};
}


std::expected<KVLayerAccess*, std::error_code> GraphContext::acquire_kv_layer(std::size_t layer) {
    if (!_kv_manager) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    auto result = _kv_manager->acquire_layer(_sequence_id, layer);
    if (!result) {
        return std::unexpected(result.error());
    }

    try {
        _kv_accesses.push_back(std::move(*result));
    } catch (...) {
        // 容器登记失败，也要归还刚刚获取的访问
        _kv_manager->release_layer(*result);
        throw;
    }

    return &_kv_accesses.back();
}
