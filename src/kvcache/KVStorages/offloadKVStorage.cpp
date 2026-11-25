#include "../../../include/kvcache/KVStorages/offloadKVStorage.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <limits>

namespace
{
    auto error(std::errc code) { 
        return std::unexpected(std::make_error_code(code)); 
    }

    struct Layout
    {
        std::size_t k_token, v_token, k_bytes, v_bytes;
        std::size_t bytes() const { return k_bytes + v_bytes; }
    };

    // 计算 K/V 占用大小
    std::expected<Layout, std::error_code> layout(const KVLayerSpec &spec, std::size_t capacity){
        auto scalar = [](DataType type){
            return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F64;
        };
        if (!scalar(spec.k_type) || !scalar(spec.v_type))
            return error(std::errc::not_supported);
        const auto limit = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
        auto size = [&](std::size_t dim, DataType type) -> std::expected<std::size_t, std::error_code>
        {
            const auto element = dataType_size(type);
            if (!capacity || !dim || !spec.n_kv_heads || !element)
                return error(std::errc::invalid_argument);
            if (dim > limit / element || spec.n_kv_heads > limit / (dim * element))
                return error(std::errc::value_too_large);
            return dim * element * spec.n_kv_heads;
        };
        auto k = size(spec.k_head_dim, spec.k_type), v = size(spec.v_head_dim, spec.v_type);
        if (!k)
            return std::unexpected(k.error());
        if (!v)
            return std::unexpected(v.error());
        if (*k > limit - *v || capacity > limit / (*k + *v))
            return error(std::errc::value_too_large);
        return Layout{*k, *v, *k * capacity, *v * capacity};
    }

    // Backend 的 set/get 接口要求 Tensor；这里只构造非拥有字节视图
    Tensor byte_view(void *data, std::size_t bytes, Backend &backend){
        Tensor view("kv_transfer", GGML_TYPE_I8, {static_cast<std::int64_t>(bytes)});
        view.data = data;
        view.backend = &backend;
        view.device = backend.device_type();
        view.offset = 0;
        return view;
    }
}

OffloadKVStorage::OffloadKVStorage(OffloadKVStorageConfig config,std::unique_ptr<KVEvictionPolicy> eviction, std::unique_ptr<KVPrefetchPolicy> prefetch)
    : _config(std::move(config))
    , _eviction(std::move(eviction))
    , _prefetch(std::move(prefetch)) 
    {}

OffloadKVStorage::~OffloadKVStorage(){
    for (const auto &[id, state] : _states){
        assert(state.active_users == 0 && "release KV accesses before destroying storage");
        destroy_object(id);
    }
}

std::expected<void, std::error_code> OffloadKVStorage::init(std::span<const KVLayerSpec> layers, std::span<Backend *const> backends){
    if (!_states.empty())
        return error(std::errc::device_or_resource_busy);
    if (layers.size() != backends.size() || !_config.host_backend || !_config.host_backend->is_host() || !_eviction || !_prefetch)
        return error(std::errc::invalid_argument);
    for (std::size_t i = 0; i < layers.size(); ++i){
        if (!backends[i])
            return error(std::errc::invalid_argument);
        auto sizes = layout(layers[i], 1);
        if (!sizes)
            return std::unexpected(sizes.error());
    }
    for (std::size_t i = 0; i < _config.device_budgets.size(); ++i){
        const auto &b = _config.device_budgets[i];
        if (!b.backend || b.backend->is_host() || b.prefetch_bytes > b.resident_bytes)
            return error(std::errc::invalid_argument);
        for (std::size_t j = 0; j < i; ++j)
            if (_config.device_budgets[j].backend == b.backend)
                return error(std::errc::invalid_argument);
    }
    for (auto *backend : backends)
    {
        if (!backend->is_host() && std::none_of(_config.device_budgets.begin(), _config.device_budgets.end(),
                                                [&](const auto &b)
                                                { return b.backend == backend; }))
            return error(std::errc::invalid_argument);
    }
    _layers.assign(layers.begin(), layers.end());
    _backend_layer.assign(backends.begin(), backends.end());
    return {};
}

std::expected<StorageId, std::error_code> OffloadKVStorage::alloc(std::size_t layer, std::size_t capacity)
{
    if (layer >= _layers.size())
        return error(std::errc::invalid_argument);
    switch (_config.first_alloc)
    {
    case KVFristAllocPlacement::CPU:
        return alloc_cpu(layer, capacity);
    case KVFristAllocPlacement::LayerBackend:
        return alloc_layerBackend(layer, capacity);
    }
    return error(std::errc::invalid_argument);
}

std::expected<StorageId, std::error_code> OffloadKVStorage::alloc_cpu(std::size_t layer, std::size_t capacity)
{
    // CPU 计算层直接由其计算 backend 分配，避免把另一 backend 的地址交给它。
    Backend *backend = _backend_layer[layer]->is_host() ? _backend_layer[layer] : _config.host_backend;
    return alloc_on(layer, capacity, *backend);
}

std::expected<StorageId, std::error_code> OffloadKVStorage::alloc_layerBackend(std::size_t layer, std::size_t capacity)
{
    return alloc_on(layer, capacity, *_backend_layer[layer]);
}

std::expected<StorageId, std::error_code> OffloadKVStorage::alloc_on(
    std::size_t layer, std::size_t capacity, Backend &backend)
{
    if (_next_storage_id == kInvalidStorageId)
        return error(std::errc::value_too_large);
    auto sizes = layout(_layers[layer], capacity);
    if (!sizes)
        return std::unexpected(sizes.error());
    if (!backend.is_host())
    {
        auto room = make_room(backend, sizes->bytes(), kInvalidStorageId);
        if (!room)
            return std::unexpected(room.error());
    }
    auto pointer = backend.alloc(BufferKind::KVCACHE, sizes->bytes());
    if (!pointer)
        return std::unexpected(pointer.error());
    if (!*pointer)
        return error(std::errc::not_enough_memory);
    const auto id = _next_storage_id++;
    KVMemoryCopy copy{&backend, *pointer, *pointer,
                      static_cast<std::byte *>(*pointer) + sizes->k_bytes, sizes->bytes(), 0};
    OffloadObjectState extra;
    (backend.is_host() ? extra.cpu : extra.gpu) = copy;
    try
    {
        _states.emplace(id, StorageState{.object = {id, &backend, capacity, copy.k_base, copy.v_base}, .layer = layer});
        _offload_states.emplace(id, std::move(extra));
    }
    catch (...)
    {
        _states.erase(id);
        _offload_states.erase(id);
        backend.free(BufferKind::KVCACHE, *pointer);
        throw;
    }
    return id;
}

// 不增加使用计数，不标记本轮 writable 有效，不提交事务
std::expected<void, std::error_code> OffloadKVStorage::prepare_object(const KVObjectAccessRequest &request)
{
    auto it = _states.find(request.id);
    auto extra = _offload_states.find(request.id);
    if (it == _states.end() || extra == _offload_states.end() || request.layer >= _layers.size() ||
        it->second.layer != request.layer || it->second.pending_free)
        return error(std::errc::invalid_argument);
    auto &state = it->second;
    if (state.active_users)
        return error(std::errc::device_or_resource_busy);
    const auto capacity = state.object.capacity_tokens;
    if (request.history_tokens > capacity || request.writable.begin > capacity ||
        request.writable.count > capacity - request.writable.begin ||
        (!request.writable.empty() && request.writable.begin != request.history_tokens))
        return error(std::errc::invalid_argument);
    auto sizes = layout(_layers[request.layer], capacity);
    if (!sizes)
        return std::unexpected(sizes.error());
    Backend *target = _backend_layer[request.layer];
    if (!target)
        return error(std::errc::invalid_argument);
    auto &copies = extra->second;
    auto &destination = target->is_host() ? copies.cpu : copies.gpu;
    // 状态由本类建立；每次复制前仍校验元数据，不能把失效地址暴露给算子。
    auto valid_memory = [&](const KVMemoryCopy &copy)
    {
        return copy.backend && copy.allocation_base && copy.k_base && copy.v_base &&
               copy.allocated_bytes >= sizes->bytes() && copy.valid_tokens <= capacity;
    };
    if (destination && (!valid_memory(*destination) || destination->backend != target))
        return error(std::errc::invalid_argument);
    auto publish = [&]()
    {
        state.object.backend = target;
        state.object.k_base = destination->k_base;
        state.object.v_base = destination->v_base;
    };
    if (destination && destination->valid_tokens >= request.history_tokens)
    {
        publish();
        return {};
    }

    const KVMemoryCopy *source = nullptr;
    const KVDiskCopy *disk = nullptr;
    if (request.history_tokens)
    {
        for (const auto *candidate : {&copies.cpu, &copies.gpu})
        {
            if (candidate == &destination || !*candidate)
                continue;
            if (!valid_memory(**candidate))
                return error(std::errc::invalid_argument);
            if ((*candidate)->valid_tokens >= request.history_tokens)
            {
                source = &**candidate;
                break;
            }
        }
        if (!source && copies.disk && copies.disk->valid_tokens >= request.history_tokens)
        {
            disk = &*copies.disk;
            const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max());
            if (disk->valid_tokens > capacity || disk->file_path.empty() ||
                disk->k_capacity_bytes < request.history_tokens * sizes->k_token ||
                disk->v_capacity_bytes < request.history_tokens * sizes->v_token ||
                disk->k_offset_bytes > limit || disk->v_offset_bytes > limit ||
                request.history_tokens * sizes->k_token > limit - disk->k_offset_bytes ||
                request.history_tokens * sizes->v_token > limit - disk->v_offset_bytes)
                return error(std::errc::invalid_argument);
        }
        if (!source && !disk)
            return error(std::errc::no_message_available);
    }

    KVMemoryCopy work;
    bool newly_allocated = !destination;
    if (destination)
        work = *destination;
    else
    {
        if (!target->is_host())
        {
            auto room = make_room(*target, sizes->bytes(), request.id);
            if (!room)
                return std::unexpected(room.error());
        }
        auto pointer = target->alloc(BufferKind::KVCACHE, sizes->bytes());
        if (!pointer)
            return std::unexpected(pointer.error());
        if (!*pointer)
            return error(std::errc::not_enough_memory);
        work = {target, *pointer, *pointer, static_cast<std::byte *>(*pointer) + sizes->k_bytes, sizes->bytes(), 0};
    }
    // 新分配在两个复制均成功后才登记；异常和错误路径均回收，已有副本保持有效前缀。
    auto cleanup = [&]()
    { if (newly_allocated) target->free(BufferKind::KVCACHE, work.allocation_base); };
    try
    {
        const auto begin = work.valid_tokens;
        const auto count = request.history_tokens - begin;
        std::ifstream file;
        if (disk && count)
        {
            file.open(disk->file_path, std::ios::binary);
            if (!file)
            {
                cleanup();
                return error(std::errc::io_error);
            }
        }
        // 磁盘使用固定上限的主机暂存；不分配一整份 CPU KV，也不改动 disk 状态。
        std::vector<std::byte> staging;
        if (disk && count)
            staging.resize(std::min<std::size_t>(1024 * 1024,
                                                 count * std::max(sizes->k_token, sizes->v_token)));
        auto transfer = [&](bool is_k) -> std::expected<void, std::error_code>
        {
            if (!count)
                return {};
            const auto token_bytes = is_k ? sizes->k_token : sizes->v_token;
            const auto offset = begin * token_bytes;
            const auto bytes = count * token_bytes;
            void *dest_base = is_k ? work.k_base : work.v_base;
            auto dest_view = byte_view(dest_base, is_k ? sizes->k_bytes : sizes->v_bytes, *target);
            auto *dest = static_cast<std::byte *>(dest_base) + offset;
            if (source)
            {
                void *src_base = is_k ? source->k_base : source->v_base;
                const auto *src = static_cast<const std::byte *>(src_base) + offset;
                if (source->backend->is_host())
                {
                    if (target->is_host())
                    {
                        std::memcpy(dest, src, bytes);
                        return {};
                    }
                    return target->copy_tensor_set(&dest_view, src, offset, bytes);
                }
                if (!target->is_host())
                    return error(std::errc::not_supported);
                auto src_view = byte_view(src_base, is_k ? sizes->k_bytes : sizes->v_bytes, *source->backend);
                return source->backend->copy_tensor_get(&src_view, dest, offset, bytes);
            }
            file.seekg(static_cast<std::streamoff>((is_k ? disk->k_offset_bytes : disk->v_offset_bytes) + offset));
            if (!file)
                return error(std::errc::io_error);
            for (std::size_t done = 0; done < bytes;)
            {
                const auto chunk = std::min(staging.size(), bytes - done);
                file.read(reinterpret_cast<char *>(staging.data()), static_cast<std::streamsize>(chunk));
                if (!file || file.gcount() != static_cast<std::streamsize>(chunk))
                    return error(std::errc::io_error);
                if (target->is_host())
                    std::memcpy(dest + done, staging.data(), chunk);
                else
                {
                    auto copied = target->copy_tensor_set(&dest_view, staging.data(), offset + done, chunk);
                    if (!copied)
                        return copied;
                }
                done += chunk;
            }
            return {};
        };
        auto k = transfer(true);
        if (!k)
        {
            cleanup();
            return k;
        }
        auto v = transfer(false);
        if (!v)
        {
            cleanup();
            return v;
        }
        work.valid_tokens = request.history_tokens;
        destination = work;
        publish();
        return {};
    }
    catch (...)
    {
        cleanup();
        throw;
    }
}

// 当前只做硬预算检查；自动选择受害者和写回尚未实现，不隐式超预算分配。
std::expected<void, std::error_code> OffloadKVStorage::make_room(
    Backend &backend, std::size_t required, StorageId)
{
    auto budget = std::find_if(_config.device_budgets.begin(), _config.device_budgets.end(),
                               [&](const auto &b)
                               { return b.backend == &backend; });
    if (budget == _config.device_budgets.end())
        return error(std::errc::invalid_argument);
    std::size_t used = 0;
    for (const auto &[id, copies] : _offload_states)
    {
        if (!copies.gpu || copies.gpu->backend != &backend)
            continue;
        if (copies.gpu->allocated_bytes > budget->resident_bytes - used)
            return error(std::errc::not_enough_memory);
        used += copies.gpu->allocated_bytes;
    }
    if (required > budget->resident_bytes - used)
        return error(std::errc::not_enough_memory);
    return {};
}

void OffloadKVStorage::destroy_object(StorageId id) noexcept
{
    auto it = _offload_states.find(id);
    if (it == _offload_states.end())
        return;
    for (auto *copy : {&it->second.cpu, &it->second.gpu})
    {
        if (*copy)
            (*copy)->backend->free(BufferKind::KVCACHE, (*copy)->allocation_base);
    }
    // 当前没有磁盘区域分配器，不删除可能由其他对象共享的文件。
    _offload_states.erase(it);
}
