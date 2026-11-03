#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <system_error>
#include <unordered_map>

#include "kvutil.hpp"
#include "../backend.hpp"
#include "../util.hpp"

struct KVStorageObject{
    StorageId id = kInvalidStorageId;
    Backend *backend = nullptr;
    std::size_t capacity_tokens = 0;
    void *k_base = nullptr;
    void *v_base = nullptr;
};

// active_users 是正在访问这份内存的次数，不是序列共享的所有权引用数
struct StorageState{
    KVStorageObject object;
    std::size_t layer = 0;
    std::size_t active_users = 0;
    bool pending_free = false;      // 这个标记是给free用的，可能有人再使用，所以等没人用的时候再释放
};

// 对象状态和使用计数由基类统一管理 当前要求调用方串行协调访问/迁移
class KVStorage
{
public:
    virtual ~KVStorage() = default;
    
    // 初始化
    virtual std::expected<void, std::error_code> init(std::span<const KVLayerSpec> layers, std::span<Backend *const> backend_layer) = 0;

    // 派生类分配后在 _states 登记 {object, layer, 0, false}，再返回稳定 ID
    // 同一 Storage 生命周期内不复用 ID
    // 失败时自行回滚尚未登记的资源
    virtual std::expected<StorageId, std::error_code> alloc(std::size_t layer, std::size_t capacity_tokens) = 0;

    // 放弃对象所有权：仍在使用则延后释放，否则立即回收
    void free(StorageId id) noexcept;
    
    // 准备到该层计算设备；成功时 active_users++，失败不增加
    // 返回普通描述，不创建访问 ID，不携带析构回调
    std::expected<KVStorageObject, std::error_code> acquire(StorageId id, std::size_t layer);

    // 计算完成后配对调用：active_users--使用计数归零才允许迁出
    // 低层调用必须严格配对，不能重复归还；上层 release_layer 会清空已归还列表
    void release_access(StorageId id) noexcept;

protected:
    // 仅在 active_users == 0 时调用。具体实现完成本地准备或迁移，更新 object
    // 返回成功时数据已就绪；不得删除该状态或改变 ID、layer、计数
    virtual std::expected<void, std::error_code> prepare_object(StorageId id, std::size_t layer) = 0;

    // 仅在计数为0时调用；归还实际内存/池块。基类随后删除 _states 条目
    virtual void destroy_object(StorageId id) noexcept = 0;

    std::unordered_map<StorageId, StorageState> _states;
    std::vector<KVLayerSpec> _layers;
    std::vector<Backend *> _backend_layer;
    StorageId _next_storage_id = 1;

};
