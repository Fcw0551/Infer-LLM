#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <tuple>

#include "../kvutil.hpp"

// Storage 提供的只读快照；策略不接触 backend、内存地址或使用计数
// candidates 已按目标设备过滤，且均可安全驱逐
struct KVEvictionCandidate{
    StorageId id = kInvalidStorageId;
    std::size_t resident_bytes = 0;
    std::size_t writeback_bytes = 0;
    std::uint64_t last_access_order = 0;
    std::uint64_t admission_order = 0;
};

class KVEvictionPolicy{
public:
    virtual ~KVEvictionPolicy() = default;

    // 一次选择一个候选；Storage 执行后重新检查预算，必要时再次选择
    // nullopt 表示没有建议；返回的 ID 必须属于 candidates
    // Storage 执行前仍需校验状态；策略不得自行写回或释放对象
    virtual std::optional<StorageId> select_victim(std::span<const KVEvictionCandidate> candidates) const = 0;
};

// 最近访问次序由 Storage 在每次成功 acquire 时维护，预取不算实际访问
class LRUKVEvictionPolicy final : public KVEvictionPolicy {
public:
    std::string name(){
        return "LRU";
    }
    std::optional<StorageId> select_victim(std::span<const KVEvictionCandidate> candidates) const override {
        if (candidates.empty())
            return std::nullopt;
        auto rank = [](const KVEvictionCandidate& candidate) {
            return std::tie(candidate.last_access_order, candidate.admission_order, candidate.id);
        };
        const auto* victim = &candidates.front();
        for (const auto& candidate : candidates.subspan(1)) {
            if (rank(candidate) < rank(*victim))
                victim = &candidate;
        }
        return victim->id;
    }
};
