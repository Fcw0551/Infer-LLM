#pragma once
#include <cstddef>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "kvutil.hpp"
#include "kvstorage.hpp"
#include "kvstrategy.hpp"

class Backend;
struct Tensor;

// 一条序列的生命周期状态
struct SequenceState {
    std::size_t committed_length = 0;   // 已提交的 token 数
    std::optional<TokenRange> pending;  // 本轮 begin_append 预留的范围
};

// 某一层 KV cache 的可写/可读视图
//
// k / v 的布局为 [head_dim, n_kv_heads, n_total, 1]：
//   - k/v 的 data 直接指向缓存内存，tensorRole 应为 TENSOR_ROLE_KVCACHE， 
//     backend 为 _backend_layer[layer]；ActivationManager 不会重复分配
//   - [0, n_past) 是已提交的历史区域，[n_past, n_total) 是本轮待写入区域
//   - 视图对象由 KVCacheManager 持有，生命周期到 commit_append / abort_append
//     或下一次 reserve_append 为止
struct KVLayerAccess {
    Tensor *k = nullptr;
    Tensor *v = nullptr;
    Backend *backend = nullptr;
    std::size_t n_past = 0;
    std::size_t n_new = 0;

    std::size_t n_total() const noexcept { return n_past + n_new; }
};

class KVCacheManager
{
public:
    KVCacheManager(std::unique_ptr<KVStorage> storage,std::unique_ptr<KVStrategy> strategy) noexcept
    : _storage(std::move(storage))
    , _strategy(std::move(strategy)) 
    {}

    ~KVCacheManager() = default;

    KVCacheManager(const KVCacheManager &) = delete;
    KVCacheManager &operator=(const KVCacheManager &) = delete;

    // 初始化缓存容量、各层计算设备、相关配置信息
    std::expected<void, std::error_code> init(std::span<const KVLayerSpec> layers,std::span<Backend *const> backend_layer,const KVCacheConfig &config);

    // 创建一条新序列，返回其身份编号  连续策略会在这里按层分配 max_tokens_per_sequence 的存储
    std::expected<SequenceId, std::error_code> create_sequence();

    // 为某条序列预留 n_tokens 个新增 token 的缓存空间
    // range = { committed_length, n_tokens }；已有 pending 时返回
    // std::errc::device_or_resource_busy；容量不足返回 std::errc::not_enough_memory
    std::expected<void, std::error_code> reserve_append(SequenceId id, std::size_t n_tokens);

    // 获取某层的 K/V 视图，包含本轮 pending 范围  要求该层已 begin_append，且 pending 非空
    std::expected<KVLayerAccess, std::error_code> acquire_layer(SequenceId id, std::size_t layer);

    // 所有层计算成功后提交本轮追加：committed_length += pending.count
    std::expected<void, std::error_code> commit_append(SequenceId id);

    // 计算失败时取消本轮追加，保留原有有效缓存
    std::expected<void, std::error_code> abort_append(SequenceId id);

    // 查询已提交的 token 数量（不含 pending）
    std::expected<std::size_t, std::error_code> length(SequenceId id) const;

    // 释放序列持有的全部缓存引用，并注销 ID
    std::expected<void, std::error_code> release_sequence(SequenceId id);

    // 清空全部序列；底层容量保留，供下次使用
    std::expected<void, std::error_code> clear();

private:
    std::unique_ptr<KVStorage> _storage;   // 物理层
    std::unique_ptr<KVStrategy> _strategy; // 逻辑层

    std::unordered_map<SequenceId, SequenceState> _sequences;
    SequenceId _next_sequence_id = 1;
};
