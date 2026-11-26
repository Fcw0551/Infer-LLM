#include "kvcache/KVStrategys/blockKVStrategy.hpp"
#include <algorithm>

namespace
{
    auto invalid() { return std::unexpected(std::make_error_code(std::errc::invalid_argument)); }
}

// 初始化
std::expected<void, std::error_code> BlockKVStrategy::init(std::span<const KVLayerSpec> layers, std::span<Backend *const> backends, const KVCacheConfig &cfg){
    if (!_sequences.empty() || !_storage || !cfg.block_tokens || !cfg.max_tokens_per_sequence || !cfg.layers || cfg.layers != layers.size() || layers.size() != backends.size())
        return invalid();
    auto r = _storage->init(layers, backends);
    if (!r)
        return r;
    _layers.assign(layers.begin(), layers.end());
    _cfg = cfg;
    return {};
}

// 注册序列
std::expected<void, std::error_code> BlockKVStrategy::create_sequence(SequenceId id){
    if (!id || !_sequences.try_emplace(id, _cfg.layers).second)
        return invalid();
    return {};
}

void BlockKVStrategy::truncate(Table &table, size_t keep) noexcept{
    for (auto &layer : table)
        while (layer.size() > keep)
        {
            _storage->free(layer.back());
            layer.pop_back();
        }
}
// 为本轮新增的位置预留空间，跨页时向底层申请缺少的页
std::expected<void, std::error_code> BlockKVStrategy::reserve_append(SequenceId id, TokenRange range){
    auto it = _sequences.find(id);
    if (it == _sequences.end())
        return invalid();
    // 检查是否超出了配置
    if (range.begin > _cfg.max_tokens_per_sequence || range.count > _cfg.max_tokens_per_sequence - range.begin)
        return std::unexpected(std::make_error_code(std::errc::not_enough_memory));
    
    // 先取出table
    auto &table = it->second;   
    
    // 向上取整计算一下需要多少页（本轮新增的 token 区间 [begin, end)）
    const size_t previous = pages(range.begin);
    const size_t needed = pages(range.end());
    try
    {
        // 分配底层空间
        for (size_t layer = 0; layer < table.size(); ++layer)
        {
            std::vector<StorageId> &ids = table[layer];
            ids.reserve(needed);
            while (ids.size() < needed)
            {
                auto block = _storage->alloc(layer, _cfg.block_tokens);
                if (!block)
                {
                    // 分配失败的时候，我们要保留之前分配的，即previous
                    truncate(table, previous);
                    return std::unexpected(block.error());
                }
                ids.push_back(*block);
            }
        }
    }
    catch (...){
        truncate(table, previous);
        throw;
    }
    return {};
}


// 确认本轮预留，将本轮刚写满的页交给底层持久化
std::expected<void, std::error_code> BlockKVStrategy::commit_append(SequenceId id, TokenRange range){
    auto it = _sequences.find(id);
    if (it == _sequences.end())
        return invalid();
    for (const auto &layer : it->second)
        for (size_t p = range.begin / _cfg.block_tokens; p < range.end() / _cfg.block_tokens; ++p)
        {
            auto r = _storage->persist(layer.at(p));
            if (!r)
                return r;
        }
    return {};
}

// 撤销本轮预留，释放新增页，保留追加前已有的尾页
std::expected<void, std::error_code> BlockKVStrategy::abort_append(SequenceId id, TokenRange range){
    auto it = _sequences.find(id);
    if (it == _sequences.end())
        return invalid();
    truncate(it->second, pages(range.begin));
    return {};
}

// 释放该序列各层的存储对象并移除页表
std::expected<void, std::error_code> BlockKVStrategy::release_sequence(SequenceId id){
    auto it = _sequences.find(id);
    if (it == _sequences.end())
        return invalid();
    truncate(it->second, 0);
    _sequences.erase(it);
    return {};
}

// 释放所有序列的存储对象并清空页表
std::expected<void, std::error_code> BlockKVStrategy::clear_sequence(){
    for (auto &[id, table] : _sequences)
        truncate(table, 0);
    _sequences.clear();
    return {};
}

// 供基类 acquire_layer 使用，按页边界把 token 区间拆成带页内偏移的 K/V 片段
std::expected<KVMapping, std::error_code> BlockKVStrategy::lookup(SequenceId id, size_t layer, TokenRange range) const{
    auto it = _sequences.find(id);
    if (it == _sequences.end() || layer >= it->second.size())
        return invalid();
    const auto &ids = it->second[layer];
    KVMapping mapping;
    for (size_t pos = range.begin; pos < range.end();)
    {
        size_t page = pos / _cfg.block_tokens, offset = pos % _cfg.block_tokens;
        if (page >= ids.size())
            return invalid();
        size_t n = std::min(range.end() - pos, _cfg.block_tokens - offset);
        mapping.k.push_back({ids[page], offset, n});
        pos += n;
    }
    mapping.v = mapping.k;
    return mapping;
}