#include "../../../include/kvcache/KVStorages/naiveKVStorage.hpp"

// 对象状态和使用计数由基类统一管理 当前要求调用方串行协调访问/迁移
// 初始化
std::expected<void, std::error_code> NaiveKVStorage::init(std::span<const KVLayerSpec> layers, std::span<Backend *const> backend_layer){
     if (layers.size() != backend_layer.size()) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    _layers.assign(layers.begin(), layers.end());
    _backend_layer.assign(backend_layer.begin(), backend_layer.end());
    return {};
}

// 派生类分配后在 _states 登记 {object, layer, 0, false}，再返回稳定 ID
// 同一 Storage 生命周期内不复用 ID
// 失败时自行回滚尚未登记的资源
std::expected<StorageId, std::error_code> NaiveKVStorage::alloc(std::size_t layer, std::size_t capacity_tokens){
    if (layer >= _layers.size()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    Backend* backend = _backend_layer[layer];
    if(!backend){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    // 计算k v 所占的内存
    KVLayerSpec& l = _layers[layer];
    std::size_t k_bytes = l.n_kv_heads*l.k_head_dim*dataType_size(l.k_type)*capacity_tokens;
    std::size_t v_bytes = l.n_kv_heads*l.v_head_dim*dataType_size(l.v_type)*capacity_tokens;

    // 进行分配即可 todo 接下来是把kv放在一起了，   K 数据 │ V 数据
    // todo 未处理对齐问题
    auto result = backend->alloc(BufferKind::KVCACHE, k_bytes + v_bytes);
    if(!result){
        return std::unexpected(result.error());
    }

    // 进行登记
    StorageId now_id = _next_storage_id++;
    KVStorageObject object = {.id = now_id, .backend = backend, .capacity_tokens = capacity_tokens, .k_base = *result, .v_base = static_cast<std::byte *>(*result) + k_bytes};
    StorageState s = {.object = object, .layer = layer, .active_users = 0, .pending_free = false};
    bool inserted = false;
    try {
        inserted = _states.try_emplace(now_id, s).second;
    } catch (...) {
        backend->free(BufferKind::KVCACHE, *result);
        throw;
    }
    if (!inserted) {
        backend->free(BufferKind::KVCACHE, *result);
        return std::unexpected(std::make_error_code(std::errc::file_exists));
    }
    return now_id;
}

// // 放弃对象所有权：仍在使用则延后释放，否则立即回收
// void free(StorageId id) noexcept;

// // 准备到该层计算设备；成功时 active_users++，失败不增加
// // 返回普通描述，不创建访问 ID，不携带析构回调
// std::expected<KVStorageObject, std::error_code> acquire(StorageId id, std::size_t layer);

// 计算完成后配对调用：active_users--使用计数归零才允许迁出
// 低层调用必须严格配对，不能重复归还；上层 release_layer 会清空已归还列表
// void release_access(StorageId id) noexcept;


// 返回成功时数据已就绪；不得删除该状态或改变 ID、layer、计数
std::expected<void, std::error_code> NaiveKVStorage::prepare_object(StorageId id, std::size_t layer){
    // 因为当前的数据本身alloc就在当前的backend，这里我们暂时做了一个朴素版本的，所以没有迁移
    if (layer >= _layers.size()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    auto it = _states.find(id);
    if (it == _states.end()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    return {};
}

// 仅在计数为0时调用,归还实际内存/池块。基类随后删除 _states 条目
void NaiveKVStorage::destroy_object(StorageId id) noexcept {
    auto it = _states.find(id);
    if (it == _states.end()){
        return;
    }
    size_t layer = it->second.layer;
    Backend* backend = _backend_layer[layer];
    if(!backend){
        return;
    }
    backend->free(BufferKind::KVCACHE,it->second.object.k_base);
}

// std::unordered_map<StorageId, StorageState> _states;
// std::vector<KVLayerSpec> _layers;
// std::vector<Backend *> _backend_layer;
// StorageID _next_storage_id = 1;
