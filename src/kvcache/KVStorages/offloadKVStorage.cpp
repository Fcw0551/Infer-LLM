#include "../../../include/kvcache/KVStorages/offloadKVStorage.hpp"

// 策略所有权移交 Storage；两者均不得为空，禁用预取使用 NoKVPrefetchPolicy
OffloadKVStorage::OffloadKVStorage(OffloadKVStorageConfig config, std::unique_ptr<KVEvictionPolicy> eviction, std::unique_ptr<KVPrefetchPolicy> prefetch)
: _config(config)
, _eviction(std::move(eviction))
,_prefetch(std::move(prefetch))
{
    std::cout << "初始化完毕，eviction 策略为" << _eviction.name() << " " << "prefetch 策略为" << _prefetch.name() << endl;
}
// 调用方必须先完成计算并归还所有访问；释放所有副本，无须保存待销毁数据
OffloadKVStorage::~OffloadKVStorage() override;

std::expected<void, std::error_code> OffloadKVStorage::init(std::span<const KVLayerSpec> layers, std::span<Backend *const> backend_layer) override{
    if (layers.size() != backend_layer.size()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    _layers.assign(layers.begin(), layers.end());
    _backend_layer.assign(backend_layer.begin(), backend_layer.end());
    return {};
}

// 只建立 CPU 后备存储并登记稳定 ID，不在 GPU 上预留每层最大容量
// CPU 计算层直接使用兼容的 CPU 存储；不得把普通主机指针标记成 GPU 指针
std::expected<StorageId, std::error_code> OffloadKVStorage::alloc(std::size_t layer, std::size_t capacity_tokens) override{
    if (layer >= _layers.size()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    switch(_config.first_alloc){
        case KVFristAllocPlacement::CPU:
            return alloc_cpu(layer,capacity_tokens);
        case KVFristAllocPlacement::LayerBackend:
            return alloc_layerBackend(layer,capacity_tokens);
        default:
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
}
std::expected<StorageId, std::error_code> OffloadKVStorage::alloc_cpu(std::size_t layer, std::size_t capacity_tokens){
    if (layer >= _layers.size()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    // 计算k v 所占的内存
    KVLayerSpec &l = _layers[layer];
    std::size_t k_bytes = l.n_kv_heads * l.k_head_dim * dataType_size(l.k_type) * capacity_tokens;
    std::size_t v_bytes = l.n_kv_heads * l.v_head_dim * dataType_size(l.v_type) * capacity_tokens;
    auto result = _config.host_backend->alloc(BufferKind::KVCACHE, k_bytes + v_bytes);
    if (!result){
        return std::unexpected(result.error());
    }
    StorageId now_id = _next_storage_id++;
    KVStorageObject object = {.id = now_id, .backend = backend, .capacity_tokens = capacity_tokens, .k_base = *result, .v_base = static_cast<std::byte *>(*result) + k_bytes};
    StorageState s = {.object = object, .layer = layer, .active_users = 0, .pending_free = false};
    bool inserted = false;   
    try{
        inserted = _states.try_emplace(now_id, s).second;
    }
    catch (...){
        backend->free(BufferKind::KVCACHE, *result);
        throw;
    }
    if (!inserted){
        backend->free(BufferKind::KVCACHE, *result);
        return std::unexpected(std::make_error_code(std::errc::file_exists));
    }
    return now_id;
}
std::expected<StorageId, std::error_code> OffloadKVStorage::alloc_layerBackend(std::size_t layer, std::size_t capacity_tokens){
    if (layer >= _layers.size()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    Backend *backend = _backend_layer[layer];
    if (!backend){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    // 计算k v 所占的内存
    KVLayerSpec &l = _layers[layer];
    std::size_t k_bytes = l.n_kv_heads * l.k_head_dim * dataType_size(l.k_type) * capacity_tokens;
    std::size_t v_bytes = l.n_kv_heads * l.v_head_dim * dataType_size(l.v_type) * capacity_tokens;

    // 进行分配即可 todo 接下来是把kv放在一起了，   K 数据 │ V 数据
    // todo 未处理对齐问题
    auto result = backend->alloc(BufferKind::KVCACHE, k_bytes + v_bytes);
    if (!result){
        return std::unexpected(result.error());
    }

    // 进行登记
    StorageId now_id = _next_storage_id++;
    KVStorageObject object = {.id = now_id, .backend = backend, .capacity_tokens = capacity_tokens, .k_base = *result, .v_base = static_cast<std::byte *>(*result) + k_bytes};
    StorageState s = {.object = object, .layer = layer, .active_users = 0, .pending_free = false};
    bool inserted = false;
    try{
        inserted = _states.try_emplace(now_id, s).second;
    }
    catch (...){
        backend->free(BufferKind::KVCACHE, *result);
        throw;
    }
    if (!inserted){
        backend->free(BufferKind::KVCACHE, *result);
        return std::unexpected(std::make_error_code(std::errc::file_exists));
    }
    return now_id;
}

// 旧基类适配位置，接入时改为接收 KVObjectAccessRequest（见文件顶部）
// 成功时 object 指向计算 backend，数据已就绪；命中驻留不重复复制
std::expected<void, std::error_code> OffloadKVStorage::prepare_object(const KVObjectAccessRequest& request) override{

}

// 基类保证 active_users == 0；销毁 CPU/GPU 全部资源及策略元信息
void OffloadKVStorage::destroy_object(StorageId id) noexcept override{

}



// 移除设备副本，保留 ID/CPU 存储；CPU-only 对象无须迁出
// 活跃访问返回 busy；写回失败保留设备副本，不允许复用其空间
// 只有 CPU 副本落后时才写回；正常追加仅需同步可能修改的尾部
std::expected<void, std::error_code> OffloadKVStorage::evict(StorageId id);

// 可选的上层调度入口：消费策略建议，第一版同步执行
// 不增加 active_users，不返回 Tensor 视图；预算不足可跳过建议
// 返回实际完成的对象数；传输失败返回错误，但已有有效副本必须保留
// 第一版预取不为投机加载驱逐其他对象，按需加载始终优先
std::expected<std::size_t, std::error_code> OffloadKVStorage::prefetch(const KVPrefetchContext &context);




// 以下为接入新基类签名/hook 时使用的内部职责边界
std::expected<void, std::error_code> OffloadKVStorage::ensure_resident(const KVObjectAccessRequest &request);
void OffloadKVStorage::record_access(const KVObjectAccessRequest &request) noexcept;

// 为按需加载腾空间。排除 protected_id、活跃对象和不安全候选；
// LRU 只选择对象，Storage 负责校验、写回和预算记账
std::expected<void, std::error_code> OffloadKVStorage::make_room(Backend &backend, std::size_t required_bytes, StorageId protected_id);

// 仅同步落后的数据；无逻辑 commit 含义
// abort 后尾部仍按未提交数据处理，下次访问不得将其作为有效历史
std::expected<void, std::error_code> OffloadKVStorage::writeback(StorageId id);
