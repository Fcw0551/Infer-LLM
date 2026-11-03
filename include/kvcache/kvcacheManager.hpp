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
    std::optional<TokenRange> pending;  // 本轮 reserve_append 预留的范围
};

class KVCacheManager
{
public:
    explicit KVCacheManager(std::unique_ptr<KVStrategy> strategy) noexcept
    : _strategy(std::move(strategy))
    {}

    ~KVCacheManager() = default;

    KVCacheManager(const KVCacheManager &) = delete;
    KVCacheManager &operator=(const KVCacheManager &) = delete;

    // 初始化缓存容量、各层计算设备、相关配置信息
    std::expected<void, std::error_code> init(std::span<const KVLayerSpec> layers,std::span<Backend *const> backend_layer,const KVCacheConfig &config);

    // 创建一条新序列，返回其身份编号
    std::expected<SequenceId, std::error_code> create_sequence();

    // 为某条序列预留 n_tokens 个新增 token 的缓存空间
    // range = { committed_length, n_tokens }；已有 pending 时返回
    // std::errc::device_or_resource_busy；容量不足返回 std::errc::not_enough_memory
    std::expected<void, std::error_code> reserve_append(SequenceId id, std::size_t n_tokens);

    // 校验序列事务后向 Strategy 请求视图,要求已有 pending
    std::expected<KVLayerAccess, std::error_code> acquire_layer(SequenceId id, std::size_t layer);

    // 等计算完成后归还访问（与 commit_append / abort_append 是不同操作）
    std::expected<void, std::error_code> release_layer(KVLayerAccess& access) noexcept;

    // 所有层计算成功后提交本轮追加：committed_length += pending.count
    std::expected<void, std::error_code> commit_append(SequenceId id);

    // 计算失败时取消本轮追加，保留原有有效缓存
    std::expected<void, std::error_code> abort_append(SequenceId id);

    // 查询已提交的 token 数量（不含 pending）
    std::expected<std::size_t, std::error_code> length(SequenceId id) const;

    // 释放序列持有的全部缓存引用，并注销 ID
    std::expected<void, std::error_code> release_sequence(SequenceId id);

    // 清空全部序列；底层容量保留，供下次使用
    std::expected<void, std::error_code> clear_sequence();

private:
    std::unique_ptr<KVStrategy> _strategy;
    std::unordered_map<SequenceId, SequenceState> _sequences;
    SequenceId _next_sequence_id = 1;
};
