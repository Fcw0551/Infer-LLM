#pragma once

// 管理序列的存储组织，并向 Manager 返回带保护的访问视图。
// 派生类实现逻辑映射；基类复用 Storage 获取和视图构造流程。
// 物理分配、驻留和迁移仍由 KVStorage 实现。
//
// 约定：
//   - 序列长度与 pending 事务状态由 KVCacheManager 唯一持有
//     Strategy 的方法都显式接收 TokenRange，自身不保存"本轮追加到哪了"
//   - reserve_append 之后，lookup 必须能看到本轮已预留的范围
//   - abort_append 撤销映射并释放对应的物理存储，长度不变
//   - 同一序列同一时刻只允许一个未提交的 append（由 Manager 保证）


// --------------- 负责维护 token → StorageId 的映射状态 ----------------------

#include <expected>
#include <memory>
#include <span>
#include <system_error>
#include <utility>
#include <vector>

#include "kvutil.hpp"
#include "kvstorage.hpp"

class KVStorage;

class KVStrategy{
public:
    explicit KVStrategy(std::unique_ptr<KVStorage> storage) noexcept
    : _storage(std::move(storage))
    {}
    virtual ~KVStrategy() = default;

    KVStrategy(const KVStrategy &) = delete;
    KVStrategy &operator=(const KVStrategy &) = delete;

    virtual std::expected<void, std::error_code> init(std::span<const KVLayerSpec> layers, std::span<Backend *const> backend_layer, const KVCacheConfig &config) = 0;

    // 注册序列
    virtual std::expected<void, std::error_code> create_sequence(SequenceId id) = 0;

    // 为本轮新增的位置预留逻辑存储
    virtual std::expected<void, std::error_code> reserve_append(SequenceId id, TokenRange range) = 0;

    // Manager 在 reserve 成功后传入 pending；begin 是历史长度，count 是本轮长度
    // 返回 [0, pending.end()) 的读视图和 pending 的写视图，持有访问保护
    // 本轮区域必须写入完成后才能读取；计算完成后必须 release_layer
    // 共同流程不需要派生类重新实现
    std::expected<KVLayerAccess, std::error_code> acquire_layer(SequenceId id, std::size_t layer, TokenRange pending);

    // 显式归还本次去重对象列表，同一个结果重复归还不重复减计数
    std::expected<void, std::error_code> release_layer(KVLayerAccess& access) noexcept;

    // 确认本轮预留；range 必须与 reserve_append 传入的一致
    virtual std::expected<void, std::error_code> commit_append(SequenceId id, TokenRange range) = 0;

    // 撤销本轮尚未提交的预留
    virtual std::expected<void, std::error_code> abort_append(SequenceId id, TokenRange range) = 0;

    // 解除序列持有的全部逻辑存储引用（映射） 以及让底层的物理存储可复用
    virtual std::expected<void, std::error_code> release_sequence(SequenceId id) = 0;

    // 清楚所有的
    virtual std::expected<void, std::error_code> clear_sequence() = 0;

protected:
    // 内部扩展点：连续/分页只决定 token 位置如何映射到存储对象 (lookup的意思就是把要找的storageID 先找出来)
    virtual std::expected<KVMapping, std::error_code> lookup(SequenceId id, std::size_t layer, TokenRange range) const = 0;
    std::expected<KVRangeAccess, std::error_code> make_KVRangeAccess(const KVMapping& mapping, TokenRange range, std::size_t layer,
                                                                    std::unordered_map<StorageId, KVStorageObject>& objects, KVLayerAccess& access);
protected:
    std::unique_ptr<KVStorage> _storage;
    std::vector<KVLayerSpec> _layers;
    KVCacheConfig _cfg = {};
};
