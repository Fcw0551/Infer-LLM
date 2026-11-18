#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <system_error>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../kvstorage.hpp"
#include "kvEvictionPolicy.hpp"
#include "kvPrefetchPolicy.hpp"

// 头文件设计稿：尚未接入现有调用链，无函数实现
// 第一版：串行协调、同步复制、CPU/GPU 搬运；磁盘仅预留位置描述，尚无 I/O 实现
// 不覆盖 acquire/release_access：现有基类的非虚接口不能被派生类替换
//
// 接入时尚需完成的调用链变动：
// 1. acquire 接收 KVObjectAccessRequest，并将其传入 prepare_object；
//    基类 prepare_object 已改声明，Naive override 和 .cpp 调用仍需同步调整；
//    不能用隐藏同名 acquire 的方式绕过
// 2. 每次成功 acquire（包括 active_users > 0 的路径）调用访问通知 hook；
//    Offload 用它更新 LRU 次序并保守登记写范围，不以 prepare 调用代替访问
// 3. KVStrategy 对同一 StorageId 汇总读写需求后只 acquire 一次
//    reserve/commit/abort 的逻辑长度仍由 Manager 管理

struct KVDeviceBudget {
    Backend* backend = nullptr; // 非拥有；一个物理设备不能重复登记独立预算
    std::size_t resident_bytes = 0; // 硬上限，0 表示不允许设备驻留
    std::size_t prefetch_bytes = 0; // 上述总预算的子集，0 表示禁用预取
};

// 首次alloc的时候是在哪里分配
enum class KVFristAllocPlacement {
    CPU,             // alloc 在 CPU 分配
    LayerBackend,    // alloc 在该层计算 backend 分配
};
struct OffloadKVStorageConfig {
    // 必须提供 CPU backend；其生命周期及预算中 backend 均长于 Storage
    Backend* host_backend = nullptr;
    // 每个非 CPU 计算设备必须有预算；包括对齐/池块造成的实际占用
    std::vector<KVDeviceBudget> device_budgets;
    // 首次alloc时，主要分配在哪里
    KVFristAllocPlacement first_alloc = KVFristAllocPlacement::CPU;
};

class OffloadKVStorage final : public KVStorage {
public:
    // 策略所有权移交 Storage；两者均不得为空，禁用预取使用 NoKVPrefetchPolicy
    OffloadKVStorage(OffloadKVStorageConfig config,std::unique_ptr<KVEvictionPolicy> eviction,std::unique_ptr<KVPrefetchPolicy> prefetch);
   
    // 调用方必须先完成计算并归还所有访问；释放所有副本，无须保存待销毁数据
    ~OffloadKVStorage() override;

    std::expected<void, std::error_code> init(std::span<const KVLayerSpec> layers,std::span<Backend *const> backend_layer) override;

    // 按 first_alloc 选择 CPU 或层计算 backend，登记稳定 ID 和初始副本
    // CPU 计算层直接使用兼容的 CPU 存储；不得把普通主机指针标记成 GPU 指针
    std::expected<StorageId, std::error_code> alloc(std::size_t layer, std::size_t capacity_tokens) override;
private:
    std::expected<StorageId, std::error_code> alloc_cpu(std::size_t layer, std::size_t capacity_tokens);
    std::expected<StorageId, std::error_code> alloc_layerBackend(std::size_t layer, std::size_t capacity_tokens);

public:
    // 移除设备副本，保留 ID/CPU 存储；CPU-only 对象无须迁出
    // 活跃访问返回 busy；写回失败保留设备副本，不允许复用其空间
    // 只有 CPU 副本落后时才写回；正常追加仅需同步可能修改的尾部
    std::expected<void, std::error_code> evict(StorageId id);

    // 可选的上层调度入口：消费策略建议，第一版同步执行
    // 不增加 active_users，不返回 Tensor 视图；预算不足可跳过建议
    // 返回实际完成的对象数；传输失败返回错误，但已有有效副本必须保留
    // 第一版预取不为投机加载驱逐其他对象，按需加载始终优先
    std::expected<std::size_t, std::error_code> prefetch(const KVPrefetchContext& context);

private:
    // 根据 request.id 查询副本表，选择覆盖所需历史的有效源；不由调用者指定源
    // 成功时 object 指向计算 backend，数据已就绪；命中驻留不重复复制
    std::expected<void, std::error_code> prepare_object(const KVObjectAccessRequest& request) override;

    // 基类保证 active_users == 0；销毁所有副本资源及策略元信息
    void destroy_object(StorageId id) noexcept override;

    // ------------ 以下为接入新基类签名/hook 时使用的内部职责边界 ----------------------
    // 保证当前对象在计算设备上可用
    std::expected<void, std::error_code> ensure_resident(const KVObjectAccessRequest& request);
    // 记录访问元信息
    void record_access(const KVObjectAccessRequest& request) noexcept;

    // 为按需加载腾空间，排除 protected_id、活跃对象和不安全候选；
    // LRU 只选择对象，Storage 负责校验、写回和预算记账
    std::expected<void, std::error_code> make_room(Backend& backend, std::size_t required_bytes, StorageId protected_id);

    // 仅同步落后的数据；无逻辑 commit 含义
    // abort 后尾部仍按未提交数据处理，下次访问不得将其作为有效历史
    std::expected<void, std::error_code> writeback(StorageId id);

    // 内存位置可描述 CPU 或 GPU；具体设备由 backend->device_type() 判定。
    // allocation_base 是唯一释放入口，k_base/v_base 可以是同一分配内的偏移。
    struct MemoryReplicaLocation {
        Backend* backend = nullptr; // 非拥有，必须比本 Storage 活得更久
        void* allocation_base = nullptr;
        void* k_base = nullptr;
        void* v_base = nullptr;
        std::size_t allocated_bytes = 0;
    };

    // 磁盘不是计算 Backend。记录 K/V 文件区域；未来由存储 I/O 层管理资源。
    // 路径仅描述位置，不代表销毁对象时可以删除整个文件（文件可能共享）。
    struct DiskReplicaLocation {
        std::string file_path;
        std::uint64_t k_offset_bytes = 0;
        std::uint64_t v_offset_bytes = 0;
        std::size_t k_bytes = 0;
        std::size_t v_bytes = 0;
    };

    struct ReplicaState {
        std::variant<MemoryReplicaLocation, DiskReplicaLocation> location;
        // 各副本采用相同 token 布局；范围是对象内坐标，排序且不重叠。
        // 仅记录已完成写入/复制、与最新数据一致的区间，分配空间不等于有效数据。
        // 写某一区间时，必须使其他副本对应区间失效；K/V 分别追踪。
        std::vector<TokenRange> valid_k;
        std::vector<TokenRange> valid_v;
    };

    struct OffloadObjectState {
        std::vector<ReplicaState> replicas;
        std::uint64_t last_access_order = 0;
        std::uint64_t admission_order = 0;
    };

    // 与 _states 使用同一个稳定 ID；不重复维护 layer、active_users、pending_free。
    // alloc 同时登记两张表（失败回滚）；destroy_object 清理本表及物理资源，
    // 基类随后删除 _states。object 仅描述当前计算视图，不是所有副本的资源所有者。
    // 迁出后应清空失效的计算地址；已 acquire 的视图存活期间禁止迁移。
    // valid_* 仅表示物理数据一致性，不表示事务提交；读取仍受 request 历史长度限制。
    // 同步第一版须在计算完成后登记新数据有效；失败/abort 的写入尾部应作废，
    // 后续接入完成通知时处理，不能仅凭 record_access 就将数据标记为有效。
    std::unordered_map<StorageId, OffloadObjectState> _offload_states;

    // 基类的成员变量
    // std::unordered_map<StorageId, StorageState> _states;
    // std::vector<KVLayerSpec> _layers;
    // std::vector<Backend *> _backend_layer;
    // StorageId _next_storage_id = 1;
    
    OffloadKVStorageConfig _config;
    std::unique_ptr<KVEvictionPolicy> _eviction;
    std::unique_ptr<KVPrefetchPolicy> _prefetch;
};
