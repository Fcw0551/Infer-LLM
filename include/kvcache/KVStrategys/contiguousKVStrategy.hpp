#pragma once

// 连续存储策略：每条序列每一层占一个
// [max_tokens_per_sequence, n_kv_heads * head_dim] 的连续存储对象
// StorageId 固定为 (sequence, layer) 对应的那一个，token_offset 就是绝对位置

// --------------- 负责维护 token → StorageId 的映射状态 ----------------------


#include "../kvstrategy.hpp"

class KVStorage;

class ContiguousKVStrategy : public KVStrategy{
public:
    explicit ContiguousKVStrategy(std::unique_ptr<KVStorage> storage) noexcept
    : KVStrategy(std::move(storage))
    {}
    ~ContiguousKVStrategy() = default;

    ContiguousKVStrategy(const ContiguousKVStrategy &) = delete;
    ContiguousKVStrategy &operator=(const ContiguousKVStrategy &) = delete;

    // 这里还要进行连续预先分配内存
    std::expected<void, std::error_code> init(std::span<const KVLayerSpec> layers, std::span<Backend *const> backend_layer, const KVCacheConfig &config);
    
    // 创建序列
    std::expected<void, std::error_code> create_sequence(SequenceId id);

    // 为本轮新增的位置预留逻辑存储
    std::expected<void, std::error_code> reserve_append(SequenceId id, TokenRange range);

    // 确认本轮预留；range 必须与 reserve_append 传入的一致
    std::expected<void, std::error_code> commit_append(SequenceId id, TokenRange range);

    // 撤销本轮尚未提交的预留
    std::expected<void, std::error_code> abort_append(SequenceId id, TokenRange range);

    // 解除序列持有的全部逻辑存储引用（映射 + 物理存储）
    std::expected<void, std::error_code> release_sequence(SequenceId id);
    
    // 清楚所有的
    std::expected<void, std::error_code> clear_sequence() override;


protected:
    // 供基类 acquire_layer 使用，外部通过访问视图接口获取缓存
    std::expected<KVMapping, std::error_code> lookup(SequenceId id, std::size_t layer, TokenRange range) const override;

private:
    std::unordered_map<SequenceId, std::vector<StorageId>> _sequences;      // vector的下标就是第几层
    // std::unique_ptr<KVStorage> _storage;
    // std::vector<KVLayerSpec> _layers;
    // KVCacheConfig _cfg = {};
};
