#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "../kvstorage.hpp"
#include "kvEvictionPolicy.hpp"
#include "kvPrefetchPolicy.hpp"

// prepare_object 和基础分配/释放已实现；完整调用链及驱逐/预取仍待接入
// 设计范围：CPU、该层的一个 GPU、磁盘；串行协调、同步搬运、连续追加
// 已支持从登记的磁盘文件区域恢复；磁盘空间管理和写出尚未实现
// 不覆盖 acquire/release_access：现有基类的非虚接口不能被派生类替换
//
// 接入时尚需完成的调用链变动：
// 1. acquire 接收 KVObjectAccessRequest，并将其传入 prepare_object；
//    基类 prepare_object 已改声明，Naive override 和 .cpp 调用仍需同步调整；
//    不能用隐藏同名 acquire 的方式绕过
// 2. 每次成功 acquire（包括 active_users > 0 的路径）调用访问通知 hook；
//    Offload 用它更新 LRU 次序；不能在 acquire 时把未完成的写入标为有效
// 3. KVStrategy 对同一 StorageId 汇总读写需求后只 acquire 一次
//    reserve/commit/abort 的逻辑长度仍由 Manager 管理
// 4. 接入计算完成/失败通知：K/V 均成功完成才推进 valid_tokens；失败或 abort
//    将所有副本前缀长度截断到保留的历史长度。现有 release_access 仅减计数，不够表达此语义
// 5. 当前 evict/writeback 声明仍描述 GPU→CPU 路径；接入磁盘时需明确源/目标层级，
//    配套磁盘预算、文件区域分配及读写接口。仅有 disk 状态不代表磁盘 offload 已可运行

struct KVDeviceBudget {
    Backend* backend = nullptr;     // 非拥有；一个物理设备不能重复登记独立预算
    std::size_t resident_bytes = 0; // 硬上限，0 表示不允许设备驻留
    std::size_t prefetch_bytes = 0; // 上述总预算的子集，0 表示禁用预取
};

// 首次alloc的时候是在哪里分配
enum class KVFristAllocPlacement {
    CPU,             // alloc 在 CPU 分配
    LayerBackend,    // alloc 在该层计算 backend 分配
};

struct OffloadKVStorageConfig {

    Backend *host_backend = nullptr;                                // 必须提供 CPU backend；其生命周期及预算中 backend 均长于 Storage

    std::vector<KVDeviceBudget> device_budgets;                     // 每个非 CPU 计算设备必须有预算；包括对齐/池块造成的实际占用

    KVFristAllocPlacement first_alloc = KVFristAllocPlacement::CPU; // 首次alloc时，主要分配在哪里
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
    friend struct OffloadKVStorageTestAccess;
    std::expected<StorageId, std::error_code> alloc_on(std::size_t layer, std::size_t capacity_tokens, Backend& backend);
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

    // 当前仅检查硬预算，不自动驱逐；不足返回 not_enough_memory。
    // 后续接入 LRU/写回时再按 protected_id 排除本次目标及活跃对象。
    std::expected<void, std::error_code> make_room(Backend& backend, std::size_t required_bytes, StorageId protected_id);

    // 仅同步落后的数据；无逻辑 commit 含义
    // abort 后尾部仍按未提交数据处理，下次访问不得将其作为有效历史
    std::expected<void, std::error_code> writeback(StorageId id);

    // 一份连续 K/V 内存。只保留一个类型，不再区分 Location 和 ReplicaState
    // allocation_base 是唯一释放入口，k_base/v_base 是其内部视图
    // 此结构不自动释放内存；释放由 Storage 统一执行，禁止复制出第二个资源所有者
    struct KVMemoryCopy {
        Backend* backend = nullptr; // 非拥有，必须比本 Storage 活得更久
        void* allocation_base = nullptr;
        void* k_base = nullptr;
        void* v_base = nullptr;
        std::size_t allocated_bytes = 0;
        // K/V 均已完成写入或复制的共同有效前缀 [0, valid_tokens)
        // 新分配为 0；不表示 Manager 已 commit，不允许超过 object.capacity_tokens
        std::size_t valid_tokens = 0;
    };

    // 一份磁盘 K/V，不属于计算 Backend，与内存副本使用相同的 token 布局
    // 用文件区域表示资源；K/V 可以在同一个共享文件的不同区域
    // Storage 负责归还所分配的区域，不能因为释放一个对象就删除整个共享文件
    struct KVDiskCopy {
        std::string file_path;
        std::uint64_t k_offset_bytes = 0;
        std::uint64_t v_offset_bytes = 0;
        std::size_t k_capacity_bytes = 0;
        std::size_t v_capacity_bytes = 0;
        // K/V 均成功写入文件的共同有效前缀；不表示事务提交或断电持久化
        // 部分写入失败不能推进该值；失败区间不得作为后续读取的数据源
        std::size_t valid_tokens = 0;
    };

    struct OffloadObjectState {
        // nullopt 表示没有这份物理存储；存在但 valid_tokens=0 表示空缓存
        std::optional<KVMemoryCopy> cpu;
        std::optional<KVMemoryCopy> gpu;
        std::optional<KVDiskCopy> disk;
        // CPU 计算层不使用 gpu，但可以有 disk；gpu.backend 必须等于该层计算 backend
        // 不额外维护 current_location/dirty，避免与指针和长度重复记录
        std::uint64_t last_access_order = 0;
        std::uint64_t admission_order = 0;
    };

    // 状态约束（串行、仅追加，已提交历史不可修改）：
    // - 稳定状态至少存在一份存储，并保留覆盖所需历史的有效源；
    //   任意副本重叠的有效前缀内容相同，不能由“有分配”推断“数据已就绪”
    // - GPU 驻留时是 GPU 层的写入目标；保留的 CPU 副本只读，可以落后
    // - cpu=100、gpu=108 时仅写回 [100,108)；cpu 不存在时写回 [0,108)
    // - disk=80、cpu=100、gpu=108 是合法状态；磁盘不要求始终包含全部数据
    //   单份驻留或多份驻留均可表达，是否保留源副本由搬运规则决定
    // - disk-only 获取到 GPU 时，经 CPU 暂存上传；暂存区不必登记成长期 cpu 副本
    // - 重试/截断前先将所有副本 valid_tokens 缩至保留的历史，不能只修改 Manager 长度
    // - 同步/计算失败不得推进有效长度；迁移成功前不得释放唯一有效源
    // - 删除副本前，必须确认其他副本覆盖其仍需保存的数据；否则先同步再释放
    //   prefetch 若迁移唯一源，也必须遵守同样的成功/失败约束
    // - 这里只固定三个存储位置，不支持多个 GPU 副本或任意历史区间改写
    //
    // 与 _states 使用同一个稳定 ID；不重复维护 layer、active_users、pending_free
    // alloc 同时登记两张表（失败回滚）；destroy_object 清理本表及物理资源，
    // 基类随后删除 _states。object 仅描述当前计算视图，不是所有副本的资源所有者
    // 迁出后应清空失效的计算地址；已 acquire 的视图存活期间禁止迁移
    // 读取仍受 request.history_tokens 限制；完成/失败通知接入点见文件顶部
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
