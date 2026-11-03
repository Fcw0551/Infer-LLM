#include "../../../include/kvcache/KVStrategys/contiguousKVStrategy.hpp"
#include "../../../include/kvcache/kvstorage.hpp"

// 连续存储策略：每条序列每一层占一个
// [max_tokens_per_sequence, n_kv_heads * head_dim] 的连续存储对象。
// StorageId 固定为 (sequence, layer) 对应的那一个，token_offset 就是绝对位置
// --------------- 负责维护 token → StorageId 的映射状态 ----------------------


// 这里还要进行连续预先分配内存
std::expected<void, std::error_code> ContiguousKVStrategy::init(std::span<const KVLayerSpec> layers, std::span<Backend *const> backend_layer, const KVCacheConfig &config){
    if (!_storage || layers.size() != backend_layer.size() || config.layers != layers.size()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    auto result = _storage->init(layers, backend_layer);
    if (!result)
        return std::unexpected(result.error());
    _layers.assign(layers.begin(), layers.end());
    _cfg = config;
    return {};
}


// 连续策略会在这里预分配每层存储
std::expected<void, std::error_code> ContiguousKVStrategy::create_sequence(SequenceId id){
    if (id == kInvalidSequenceId){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (_sequences.contains(id)){
        return std::unexpected(std::make_error_code(std::errc::file_exists));
    }
    std::vector<StorageId> tmp;
    tmp.reserve(_cfg.layers);
    auto rollback = [&]() noexcept {
        for (auto storage_id : tmp){
            _storage->free(storage_id);
        }
    };

    try {
        for(size_t i=0;i<_cfg.layers;++i){
            auto storageId = _storage->alloc(i, _cfg.max_tokens_per_sequence);
            if (!storageId){
                rollback();
                return std::unexpected(storageId.error());
            }
            tmp.push_back(*storageId);
        }
        // 先登记空映射，成功后再转移 ID，保证插入抛异常时 tmp 仍可用于回滚
        auto [it, inserted] = _sequences.try_emplace(id);
        if (!inserted){
            rollback();
            return std::unexpected(std::make_error_code(std::errc::file_exists));
        }
        it->second = std::move(tmp);
    } catch (...) {
        rollback();
        throw;
    }
    return {};
}

// 为本轮新增的位置预留逻辑存储
std::expected<void, std::error_code> ContiguousKVStrategy::reserve_append(SequenceId id, TokenRange range){
    if (id == kInvalidSequenceId){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if(!_sequences.contains(id)){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    const auto capacity = _cfg.max_tokens_per_sequence;

    // todo：现在超出了之间报错，不截断，不覆盖历史不自动滑窗
    if (range.begin > capacity || range.count > capacity - range.begin){
        return std::unexpected(std::make_error_code(std::errc::not_enough_memory));
    }

    return {};
}

// 查询某层某段位置的 K/V 映射，包含本轮已预留的位置
std::expected<KVMapping, std::error_code> ContiguousKVStrategy::lookup(SequenceId id, std::size_t layer, TokenRange range) const{
    if (id == kInvalidSequenceId){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if(!_sequences.contains(id)){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    
    const auto &cur = _sequences.at(id);
    if (layer >= cur.size()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    const auto capacity = _cfg.max_tokens_per_sequence;
    if (range.begin > capacity || range.count > capacity - range.begin){
        return std::unexpected(std::make_error_code(std::errc::not_enough_memory));
    }
    if (range.empty()){
        return KVMapping{};
    }
    KVSlice k_slice = {.storage_id=cur[layer],.token_offset=range.begin,.token_count=range.count};
    KVSlice v_slice = {.storage_id=cur[layer],.token_offset=range.begin,.token_count=range.count};

    std::vector<KVSlice> k{k_slice};
    std::vector<KVSlice> v{v_slice};

    return KVMapping{.k=std::move(k),.v=std::move(v)};
}

// 确认本轮预留；range 必须与 reserve_append 传入的一致
std::expected<void, std::error_code> ContiguousKVStrategy::commit_append(SequenceId id, TokenRange range){
    if(!_sequences.contains(id)){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    const auto capacity = _cfg.max_tokens_per_sequence;
    if (range.begin > capacity || range.count > capacity - range.begin){
        return std::unexpected(std::make_error_code(std::errc::not_enough_memory));
    }
    return {};
}

// 撤销本轮尚未提交的预留
std::expected<void, std::error_code> ContiguousKVStrategy::abort_append(SequenceId id, TokenRange range){
    if(!_sequences.contains(id)){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    const auto capacity = _cfg.max_tokens_per_sequence;
    if (range.begin > capacity || range.count > capacity - range.begin){
        return std::unexpected(std::make_error_code(std::errc::not_enough_memory));
    }
    return {};
}

// 解除序列持有的全部逻辑存储引用（映射 + 物理存储）
std::expected<void, std::error_code> ContiguousKVStrategy::release_sequence(SequenceId id){
    if(!_sequences.contains(id)){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    std::vector<StorageId>& cur = _sequences.at(id);
    for(auto& s_id : cur){
        _storage->free(s_id);
    }
    _sequences.erase(id);
    return {};
}

// 清楚所有的
std::expected<void, std::error_code> ContiguousKVStrategy::clear_sequence(){
    for (auto &[id, storage_ids] : _sequences){
        for (auto s_id : storage_ids){
            _storage->free(s_id);
        }
    }
    _sequences.clear();
    return {};
}
