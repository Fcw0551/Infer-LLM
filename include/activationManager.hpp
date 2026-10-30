#pragma once 
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstddef>
#include <expected>
#include <system_error>

class Tensor;
class Backend;
class Graph;

// 与内存池不一样，这里主要是计算图的内存的规划者


// 记录每个backend开辟了多少，后续需要释放
struct AllocationInfo{
    Backend *backend = nullptr;
    void *ptr = nullptr;
    std::size_t size = 0;
};

// 每个backend要开多大的内存
struct BackendMemoryPlan {
    Backend* backend = nullptr;
    size_t required_bytes = 0;
};

// plan的产出，每个tensor应该如何分配
struct TensorPlacementPlan {
    Tensor* tensor = nullptr;  // 借用，只用于绑定
    std::size_t block_index = 0;
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

class ActivationManager{
public:
    ActivationManager(std::vector<Backend *> backends)
    : _backends(backends)
    {}
    ~ActivationManager(){
       release();
    }
    ActivationManager(const ActivationManager &) = delete;
    ActivationManager &operator=(const ActivationManager &) = delete;
    // 规划计算图
    std::expected<void, std::error_code> plan(const Graph& g);
    // 申请内存；分配异常转换为错误码，不向调用方传播
    // Backend::free 必须遵守不抛异常的资源回收约定
    std::expected<void, std::error_code> allocate() noexcept;
    // bind tensor->data
    std::expected<void, std::error_code> bind();
    // 清除本轮规划和 Tensor 引用，保留 Buffer 供下一轮复用；可重复调用
    // 调用前必须完成设备操作及输出读取。旧 Tensor::data 不再有效，且不会被置空
    // 不解引用旧 Tensor，因此允许 Graph 先销毁
    std::expected<void, std::error_code> reset();
    // 清除规划并释放持有的 Buffer；可重复调用，不访问旧 Tensor
    // Backend 必须仍然存活，设备操作及输出读取必须已完成
    std::expected<void, std::error_code> release();
private:
    struct ActiveRegion {
        std::size_t offset;
        std::size_t bytes;
        std::size_t last_use;
    };
    struct FreeList {
        std::size_t offset;
        std::size_t bytes;
    };
    struct BackendState {
        std::size_t block_index = 0;
        std::size_t arena_bytes = 0;
        std::vector<ActiveRegion> active;
        std::vector<FreeList> free_list;
    };

    std::expected<bool, std::error_code> need_alloc(Tensor *t);
    std::expected<BackendState *, std::error_code> get_backend_state(std::unordered_map<Backend *, BackendState> &states, Backend *backend);
    void recycle_finished_regions(BackendState &state, std::size_t i);
    std::expected<void, std::error_code> merge_free_list(std::vector<FreeList> &free_list);
    std::expected<std::size_t, std::error_code> find_or_extend(BackendState &state, std::size_t bytes);

private:
    // plan的产出
    struct AllocationPlan
    {
        std::vector<BackendMemoryPlan> blocks;                  // 记录backend开多大空间
        std::vector<TensorPlacementPlan> placements;            // 记录每个tensor放在哪个blocks，总共需要多少的内存
    };
    AllocationPlan _plan;                                       // plan的产出
    std::unordered_map<Tensor*,std::size_t> last_uses;          // 记录当前tensor最后被使用的下标
    std::vector<Backend *> _backends;
    std::vector<AllocationInfo> _allocations;                   // 后续需要释放的
};
