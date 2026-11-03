#pragma once

// KV cache 模块共享的基础类型

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <utility>

#include "../util.hpp"
#include "../tensor.hpp"

class KVStrategy;
using StorageId = std::uint64_t;
constexpr StorageId kInvalidStorageId = 0;

// 序列 ID：一条请求（一次生成）对应一个
using SequenceId = std::uint64_t;

constexpr SequenceId kInvalidSequenceId = 0;

// 每层 KV cache 的形状与数据类型
struct KVLayerSpec
{
    std::size_t n_kv_heads = 0; // KV 头数（GQA/MQA）
    std::size_t k_head_dim = 0; // 单个 K 头维度
    std::size_t v_head_dim = 0; // 单个 V 头维度

    DataType k_type = GGML_TYPE_F32;
    DataType v_type = GGML_TYPE_F32;
};

// 缓存容量与布局配置
struct KVCacheConfig{
    // model一共有多少层
    std::size_t layers = 0;
    // 单条序列最多缓存多少 token（单层）
    std::size_t max_tokens_per_sequence = 0;

    // 块大小；0 表示不切块（v1 连续策略：每条序列每层一整段）
    std::size_t block_tokens = 0;

    // 物理起始地址对齐（SIMD / CUDA 友好）
    std::size_t alignment = 256;
};

// token 位置区间 [begin, begin + count)
// 语义上和 "绝对位置" 保持一致：序列内已提交 token 从 0 开始编号
struct TokenRange
{
    std::size_t begin = 0;
    std::size_t count = 0;

    [[nodiscard]] std::size_t end() const noexcept { return begin + count; }
    [[nodiscard]] bool empty() const noexcept { return count == 0; }
};

// 一段 token 在某个存储对象内的位置
struct KVSlice
{
    StorageId storage_id = kInvalidStorageId;
    std::size_t token_offset = 0; // 在该存储对象内从哪个 token 开始
    std::size_t token_count = 0;  // 此片段包含多少个 token
};

// 一次查询的结果，按查询 token 的顺序排列
// K / V 分开是为了后续支持不同的存储布局或量化类型
struct KVMapping
{
    std::vector<KVSlice> k;
    std::vector<KVSlice> v;
};


// 一段 Tensor 的非拥有视图，不负责释放缓存内存
struct KVSegmentView {
    Tensor view;                  // data 已定位到这一段的起点
    std::size_t token_begin = 0;   // 在序列中的逻辑起点
    std::size_t token_count = 0;
};

// 一组按逻辑 token 顺序排列的 K/V 分段
struct KVRangeAccess {
    std::vector<KVSegmentView> k;
    std::vector<KVSegmentView> v;
};


struct KVLayerAccess{
private:
    friend class KVStrategy;
    // 本次成功 acquire 的去重对象列表；release_layer 逐个归还后清空
    std::vector<StorageId> _storage_ids;
    KVStrategy* _owner = nullptr;

public:
    KVLayerAccess() = default;
    KVLayerAccess(const KVLayerAccess &) = delete;
    KVLayerAccess &operator=(const KVLayerAccess &) = delete;
    KVLayerAccess(KVLayerAccess &&other) noexcept
        : _storage_ids(std::move(other._storage_ids)),
          _owner(std::exchange(other._owner, nullptr)),
          backend(std::exchange(other.backend, nullptr)),
          n_past(std::exchange(other.n_past, 0)),
          n_new(std::exchange(other.n_new, 0)),
          readable(std::move(other.readable)), writable(std::move(other.writable))
    {
        other._storage_ids.clear();
        other.readable.k.clear(); other.readable.v.clear();
        other.writable.k.clear(); other.writable.v.clear();
    }
    // 不允许覆盖一个尚未归还的访问。移动构造可用于返回结果和转交执行器。
    KVLayerAccess& operator=(KVLayerAccess&&) = delete;
    ~KVLayerAccess() {
        // 只检查漏归还，不自动减计数，也不自动等待 GPU。
        assert(_owner == nullptr && "call release_layer after computation completes");
    }

    // 计算完成后必须显式调用同一 Manager/Strategy 的 release_layer。
    // release_layer 后视图失效；失败和取消路径也必须归还。
    Backend *backend = nullptr;

    std::size_t n_past = 0;
    std::size_t n_new = 0;

    KVRangeAccess readable; // [0, n_past + n_new)
    KVRangeAccess writable; // [n_past, n_past + n_new)
};
