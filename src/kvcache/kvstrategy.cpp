#include "../../include/kvcache/kvstrategy.hpp"
#include "../../include/kvcache/kvstorage.hpp"
#include "../../include/backend.hpp"
#include <limits>


std::expected<KVLayerAccess, std::error_code> KVStrategy::acquire_layer(SequenceId id, std::size_t layer, TokenRange pending) {
    if (!_storage || layer >= _layers.size()) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    const size_t n_past = pending.begin;
    const size_t n_new = pending.count;
    if (n_new > std::numeric_limits<size_t>::max() - n_past) {
        return std::unexpected(std::make_error_code(std::errc::value_too_large));
    }

    const TokenRange read_range{.begin = 0, .count = n_past + n_new};
    const TokenRange write_range = pending;
    
    auto read_mapping = lookup(id, layer, read_range);  
    if (!read_mapping){
        return std::unexpected(read_mapping.error());
    }

    auto write_mapping = lookup(id, layer, write_range);
    if (!write_mapping){
        return std::unexpected(write_mapping.error());
    }

    KVLayerAccess access;
    access._owner = this;
    access.n_past = n_past;
    access.n_new = n_new;
    std::unordered_map<StorageId, KVStorageObject> objects;
    // 尚未交给外部执行；所有失败路径可立即归还已经增加的计数
    try {
        auto readable = make_KVRangeAccess(*read_mapping, read_range, layer, objects, access);
        if (!readable) {
            release_layer(access);
            return std::unexpected(readable.error());
        }
        auto writable = make_KVRangeAccess(*write_mapping, write_range, layer, objects, access);
        if (!writable) {
            release_layer(access);
            return std::unexpected(writable.error());
        }
        access.readable = std::move(*readable);
        access.writable = std::move(*writable);
        if (!objects.empty()) access.backend = objects.begin()->second.backend;
        return access;
    } catch (...) {
        release_layer(access);
        throw;
    }
}

std::expected<void, std::error_code> KVStrategy::release_layer(KVLayerAccess& access) noexcept {
    if (!access._owner) return {}; // 已归还，或已移动到另一结果
    if (access._owner != this) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    for (StorageId id : access._storage_ids) _storage->release_access(id);
    access._storage_ids.clear();
    access._owner = nullptr;
    access.readable.k.clear(); access.readable.v.clear();
    access.writable.k.clear(); access.writable.v.clear();
    access.backend = nullptr;
    access.n_past = access.n_new = 0;
    return {};
}

// 映射解析和视图构造复用原 Manager 实现
std::expected<KVRangeAccess, std::error_code> KVStrategy::make_KVRangeAccess(const KVMapping& mapping, TokenRange range, std::size_t layer,std::unordered_map<StorageId, KVStorageObject>& objects, KVLayerAccess& access) {
    // 校验
    if (!_storage || layer >= _layers.size()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    const auto &spec = _layers[layer];
    // K/V 和读写范围按 StorageId 去重，每个对象本次只增加一次计数（防止同一段的内存被多少++）
    Backend *backend = objects.empty() ? nullptr : objects.begin()->second.backend;

    auto build = [&](const std::vector<KVSlice> &slices, bool is_k) -> std::expected<std::vector<KVSegmentView>, std::error_code> {
        const auto type = is_k ? spec.k_type : spec.v_type;
        const auto head_dim = is_k ? spec.k_head_dim : spec.v_head_dim;
        // 当前 Tensor 的 stride 构造仅适用于逐元素类型；量化块需另行描述布局 TODO
        if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16 && type != GGML_TYPE_F64){
            return std::unexpected(std::make_error_code(std::errc::not_supported));
        }
        const auto element_bytes = static_cast<std::size_t>(dataType_size(type));
        const auto limit = static_cast<std::size_t>(std::numeric_limits<int64_t>::max());
        
        // 校验
        if (head_dim == 0 || spec.n_kv_heads == 0 || element_bytes == 0){
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        }
        if (head_dim > limit / element_bytes || spec.n_kv_heads > limit / (head_dim * element_bytes)){
            return std::unexpected(std::make_error_code(std::errc::value_too_large));
        }

        // 计算单个token的字节数
        const auto token_bytes = head_dim * spec.n_kv_heads * element_bytes;
        std::vector<KVSegmentView> segments;
        segments.reserve(slices.size());
        std::size_t covered = 0;                // 用于记录slice要写多少个token  KVSegmentView要对应
        
        // 遍历每个小片段，然后产出即可
        for (const auto &slice : slices){
            if (slice.storage_id == kInvalidStorageId || slice.token_count == 0 || slice.token_count > range.count - covered){
                return std::unexpected(std::make_error_code(std::errc::invalid_argument));
            }
            // 找一下之前有没有计数过，如果有就不计数，没有就开始计数
            auto it = objects.find(slice.storage_id);
            if (it == objects.end()){
                // 本次首次访问这个对象：准备数据并增加使用计数
                auto acquired = _storage->acquire(slice.storage_id, layer);
                if (!acquired)
                    return std::unexpected(acquired.error());
                try {
                    access._storage_ids.push_back(slice.storage_id);
                } catch (...) {
                    _storage->release_access(slice.storage_id);
                    throw;
                }
                it = objects.emplace(slice.storage_id, *acquired).first;        // 返回迭代器对象
            }
            const auto &object = it->second;
            void *base = is_k ? object.k_base : object.v_base;                  // 判断当前拿的是k的还是v的
            if (object.id != slice.storage_id || !base || !object.backend ||
                slice.token_offset > object.capacity_tokens ||
                slice.token_count > object.capacity_tokens - slice.token_offset){
                return std::unexpected(std::make_error_code(std::errc::invalid_argument));
            }
            
            if (backend && backend != object.backend) {
                return std::unexpected(std::make_error_code(std::errc::invalid_argument));
            }
            backend = object.backend;

            Tensor view(is_k ? "kv_k_view" : "kv_v_view", type,
                        {static_cast<int64_t>(head_dim),
                         static_cast<int64_t>(spec.n_kv_heads),
                         static_cast<int64_t>(slice.token_count), 1});
            view.data = static_cast<std::byte *>(base) + slice.token_offset * token_bytes;  // 计算片段起始的偏移量
            view.offset = 0;                                                                // data 已经定位到片段起点
            view.layer = static_cast<int>(layer);
            view.backend = object.backend;
            view.device = object.backend->device_type();
            view.tensorRole = TensorRole::TENSOR_ROLE_KVCACHE;                                                          
            segments.push_back(KVSegmentView{
                .view = std::move(view),
                .token_begin = range.begin + covered,
                .token_count = slice.token_count});
            covered += slice.token_count;
        }
        if (covered != range.count)
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        return segments;
    };

    auto k = build(mapping.k, true);
    if (!k)
        return std::unexpected(k.error());
    auto v = build(mapping.v, false);
    if (!v)
        return std::unexpected(v.error());
    return KVRangeAccess{.k = std::move(*k), .v = std::move(*v)};
}
