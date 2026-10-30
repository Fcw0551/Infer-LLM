#include "../include/activationManager.hpp"
#include "../include/graph.hpp"
#include "../include/backend.hpp"
#include <algorithm>
#include <new>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
// 0/default: off; 1: per-device summary; 2: include per-tensor placements.
int activation_debug_level() noexcept {
    static const int level = [] {
        const char* value = std::getenv("INFERLLM_ACTIVATION_DEBUG");
        if (!value) return 0;
        if (std::strcmp(value, "2") == 0) return 2;
        return std::strcmp(value, "1") == 0 ? 1 : 0;
    }();
    return level;
}
}
// 与内存池不一样，这里主要是计算图的内存的规划者

// 规划计算图
std::expected<void, std::error_code> ActivationManager::plan(const Graph &g){
    // kv tensor     底层是由kvmanager管理
    // weight tensor 底层是由model管理
    // 外部输入       底层是由最外层管理
    // 注意以上的为0也不管，因为底层的空间都不是这里管理的，这里只管理普通的activation

    // TODO:暂时没有考虑原地计算,view等特殊的

    _plan = {};

    // 倒着构建最后一个所使用的
    last_uses.clear();
    const auto &nodes = g.nodes();
    for (std::size_t i = nodes.size(); i > 0;){
        --i;
        Tensor *node = nodes[i];

        // 没有消费者时，至少保留到本节点执行完成
        // 已被后续节点使用过，则保留已记录的更晚位置
        last_uses.try_emplace(node, i);

        for (Tensor *src : node->src){
            if (src)
                last_uses.try_emplace(src, i);
        }
    }
    // 输出在本轮规划中不回收
    for (Tensor *output : g.outputs())
        last_uses[output] = nodes.size();   // 取一个较大的数，防止后面plan的时候，内存被复用

    // A(0,1) B(1,2) B_copy(2,3) C(3,5) d(4,5) e(5,-1)
    // 一个tensor的生存空间应该是 [ 自己的下标，最后消费的位置last_use ]
    
    // 每个 Backend 对应一个固定的 block_index；重复 Backend 不重复建块
    std::unordered_map<Backend*, BackendState> states;
    states.reserve(_backends.size());
    for (Backend* backend : _backends) {
        if (!backend)
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        auto [it, inserted] = states.try_emplace(backend);
        if (inserted){
            it->second.block_index = _plan.blocks.size();
            _plan.blocks.push_back({backend, 0});
        }
    }

    // 正向遍历，规划状态
    for (std::size_t i = 0; i < nodes.size(); ++i){
        Tensor *t = nodes[i];

        // 外部已有存储、权重、KV cache 等不由这里分配
        auto r = need_alloc(t);
        if (!r)
            return std::unexpected(r.error());
        if (!(*r))
            continue;
        
        // 拿到当前的backend的状态
        auto state_result = get_backend_state(states, t->backend);
        if (!state_result)
            return std::unexpected(state_result.error());
        BackendState& state = **state_result;

        // i是当前遍历到的区域，我们需要把之前的分配出去的先检查能不能回收到free_list里面
        // 如果一个激活区域的last_use < i 才能被回收，等于不行，那证明当前i要读取，不能回收
        recycle_finished_regions(state, i);

        // 合并相邻空闲区域，方便容纳更大的张量
        if (auto r = merge_free_list(state.free_list); !r)
            return std::unexpected(r.error());

        // 优先从空闲区域取空间；没有合适的，就从尾部扩展
        const std::size_t previous_arena_bytes = state.arena_bytes;
        auto offset_result = find_or_extend(state, t->byte_size());
        if (!offset_result)
            return std::unexpected(offset_result.error());
        std::size_t offset = *offset_result;
        if (activation_debug_level() >= 2) {
            std::fprintf(stderr,
                "[activation][placement] backend=%s tensor=%s bytes=%zu offset=%zu last_use=%zu action=%s\n",
                t->backend->get_info().name.c_str(), t->name.c_str(), t->byte_size(),
                offset, last_uses.at(t), t->byte_size() == 0 ? "EMPTY" :
                state.arena_bytes == previous_arena_bytes ? "FREE_LIST" : "EXTEND");
        }

        // 保存这个 Tensor 的布局,也就是plan的产出
        _plan.placements.push_back({
            t,
            state.block_index,
            offset,
            t->byte_size(),
        });

        // 标记这段空间在最后一次使用之前不能复用。
        state.active.push_back({
            offset,
            t->byte_size(),
            last_uses.at(t),
        });
    }
    // 保存各设备的最终峰值容量，供 allocate() 申请底层 Buffer
    for (const auto& [backend, state] : states) {
        _plan.blocks[state.block_index].required_bytes = state.arena_bytes;
    }
    if (activation_debug_level()) {
        for (std::size_t i = 0; i < _plan.blocks.size(); ++i) {
            long double tensor_bytes = 0;
            std::size_t tensors = 0;
            for (const auto& placement : _plan.placements) {
                if (placement.block_index == i) {
                    tensor_bytes += placement.bytes;
                    ++tensors;
                }
            }
            const auto& block = _plan.blocks[i];
            // 比较逐 Tensor 独立分配的数据字节总量；Arena 包含对齐填充。
            std::fprintf(stderr,
                "[activation][plan] backend=%s tensors=%zu tensor_bytes=%.0Lf arena_bytes=%zu reduction=%.2Lf%%\n",
                block.backend->get_info().name.c_str(), tensors, tensor_bytes,
                block.required_bytes, tensor_bytes == 0 ? 0.0L :
                100.0L * (1.0L - block.required_bytes / tensor_bytes));
        }
    }
    return {};
}

// 申请内存
std::expected<void, std::error_code> ActivationManager::allocate() noexcept{
    // 调用前，上一轮设备操作和输出读取必须已经完成
    // 一台 Backend 对应一个块，先检查计划，避免重复复用同一块
    for (std::size_t i = 0; i < _plan.blocks.size(); ++i) {
        if (!_plan.blocks[i].backend)
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        for (std::size_t j = 0; j < i; ++j) {
            if (_plan.blocks[j].backend == _plan.blocks[i].backend)
                return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        }
    }

    // 只记录本次新申请的块；失败时自动回滚，旧块保持不变
    struct PendingAllocations {
        std::vector<AllocationInfo> blocks;
        ~PendingAllocations() {
            for (const auto& block : blocks)
                block.backend->free(BufferKind::ACTIVATION, block.ptr);
        }
    } pending;

    // try主要是捕捉vector等分配的异常
    try {
        std::vector<AllocationInfo> next_allocations;
        std::vector<bool> retained(_allocations.size(), false);
        // 提前预留，确保申请成功后登记指针不会触发容器扩容。
        next_allocations.reserve(_plan.blocks.size());
        pending.blocks.reserve(_plan.blocks.size());

        // 在当前的_plan中，找到backend是属于_allocations的哪个下标
        for (const auto& block : _plan.blocks) {
            auto it = std::find_if(_allocations.begin(), _allocations.end(),[&](const AllocationInfo& allocation) {
                                                                            return allocation.backend == block.backend;}
                                );
            
            // 如果之前开过的空间足够容纳，则复用
            if (it != _allocations.end() && it->size >= block.required_bytes && (it->ptr || it->size == 0)) {
                // 容量足够就复用，保留实际容量，并按本轮计划排列
                next_allocations.push_back(*it);
                retained[static_cast<std::size_t>(it - _allocations.begin())] = true;   // 标记可以
                continue;
            }

            // 零容量保留占位，保证 block_index 一致
            if (block.required_bytes == 0) {
                next_allocations.push_back({block.backend, nullptr, 0});
                continue;
            }

            // 说明原先的容量不够了，需要申请
            auto result = block.backend->alloc(BufferKind::ACTIVATION, block.required_bytes);
            if (!result)
                return std::unexpected(result.error());
            if (!*result)
                return std::unexpected(std::make_error_code(std::errc::not_enough_memory));

            const AllocationInfo allocation{block.backend, *result, block.required_bytes};
            pending.blocks.push_back(allocation);
            next_allocations.push_back(allocation);
        }

        // 全部成功才提交；失败不会改变旧记录的顺序、容量和指针
        _allocations.swap(next_allocations);
        pending.blocks.clear(); // 新块所有权已交给 _allocations

        // 仅在提交成功后打印，避免把随后回滚的临时申请误报为跨轮扩容
        if (activation_debug_level()){
            for (std::size_t i = 0; i < _allocations.size(); ++i){
                const auto &allocation = _allocations[i];
                auto old = std::find_if(next_allocations.begin(), next_allocations.end(),[&](const AllocationInfo &a){ return a.backend == allocation.backend; });
                const bool reused = old != next_allocations.end() && retained[static_cast<std::size_t>(old - next_allocations.begin())];
                std::fprintf(stderr,"[activation][buffer] backend=%s action=%s required=%zu old_capacity=%zu capacity=%zu old_ptr=%p ptr=%p\n",
                             allocation.backend->get_info().name.c_str(), !allocation.ptr ? "EMPTY" : reused
                             ? "REUSE": old != next_allocations.end() && old->ptr 
                             ? "GROW" : "NEW",
                             _plan.blocks[i].required_bytes,
                             old == next_allocations.end() ? std::size_t{0} : old->size,
                             allocation.size, old == next_allocations.end() ? nullptr : old->ptr,
                             allocation.ptr);
            }
        }

        // next_allocations 此时是旧记录，释放被替换或不再需要的块
        for (std::size_t i = 0; i < next_allocations.size(); ++i) {
            const auto& old = next_allocations[i];
            if (!retained[i] && old.ptr)
                old.backend->free(BufferKind::ACTIVATION, old.ptr);
        }
        return {};
    } catch (const std::bad_alloc&) {
        return std::unexpected(std::make_error_code(std::errc::not_enough_memory));
    } catch (const std::length_error&) {
        return std::unexpected(std::make_error_code(std::errc::value_too_large));
    } catch (const std::system_error& error) {
        return std::unexpected(error.code());
    } catch (...) {
        // Backend 的其他分配异常也转换为错误码；pending 回收本次新块
        return std::unexpected(std::make_error_code(std::errc::io_error));
    }
}
// bind tensor->data
std::expected<void, std::error_code> ActivationManager::bind(){
    // 验证
    if (_allocations.size() != _plan.blocks.size())
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));

    // 绑定
    for (const auto &placement : _plan.placements){
        const auto &allocation = _allocations[placement.block_index];
        placement.tensor->data = placement.bytes == 0 ? nullptr : static_cast<std::byte *>(allocation.ptr) + placement.offset;
    }
    return {};
}
// 只结束本轮规划，保留实际分配的容量用于跨轮复用
std::expected<void, std::error_code> ActivationManager::reset(){
    // 图可能已经销毁，仅丢弃指针记录，不能通过 placements 访问 Tensor
    _plan.placements.clear();
    _plan.blocks.clear();
    last_uses.clear();
    return {};
}

// 释放所有拥有的底层块，不逐 Tensor 释放（多个 Tensor 可能共享区域）
std::expected<void, std::error_code> ActivationManager::release(){
    reset();
    for (const auto& allocation : _allocations) {
        if (allocation.ptr)
            allocation.backend->free(BufferKind::ACTIVATION, allocation.ptr);
    }
    _allocations.clear();
    return {};
}













// -------------------------------------- helper ------------------------------------------
std::expected<bool, std::error_code> ActivationManager::need_alloc(Tensor *t){
    // 外部已有存储、权重、KV cache 等不由这里分配
    if(t->tensorRole==TensorRole::TENSOR_ROLE_ACTIVATION){
        return true;
    }
    else{
        return false;
    }
}
// 拿到当前的backend的状态
std::expected<ActivationManager::BackendState*, std::error_code> ActivationManager::get_backend_state(std::unordered_map<Backend*, BackendState>& states, Backend* b){
    auto it = states.find(b);
    if (it == states.end())
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    return &it->second;
}

// i是当前遍历到的区域，我们需要把之前的分配出去的先检查能不能回收到free_list里面
// 如果一个激活区域的last_use < i 才能被回收，等于不行，那证明当前i要读取，不能回收
void ActivationManager::recycle_finished_regions(BackendState& state, std::size_t i){
    for (auto it = state.active.begin(); it != state.active.end();) {
        // 先记录空闲区域，再删除  last_use == i 的区域仍需要保留
        if (it->last_use < i) {
            state.free_list.push_back({it->offset, it->bytes});
            it = state.active.erase(it);
        } else {
            ++it;
        }
    }
}

// 按地址排序后合并相邻区域，重叠说明存在重复回收，不能静默合并
std::expected<void, std::error_code> ActivationManager::merge_free_list(std::vector<FreeList>& free_list){
    // size_t 是无符号类型：相加后变小，说明结果发生了回绕
    // 先检查每个区间的终点，后面的排序、相邻判断才可以安全使用加法
    for (const auto& region : free_list) {
        const std::size_t end = region.offset + region.bytes;
        if (end < region.offset)
            return std::unexpected(std::make_error_code(std::errc::value_too_large));
    }

    // 遍历所有的free_list,如果里面的bytes==0 直接删掉
    std::erase_if(free_list, [](const FreeList& region) { return region.bytes == 0; });
    
    // 按照offset排个序先
    std::sort(free_list.begin(), free_list.end(), [](const FreeList& a, const FreeList& b) {
        return a.offset < b.offset;
    });

    // 先验证，再合并，避免发现重叠时已经部分压缩了列表
    for (std::size_t i = 1; i < free_list.size(); ++i) {
        const auto& prev = free_list[i - 1];
        if (prev.offset + prev.bytes > free_list[i].offset)
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    // 如果采用erase会导致每次都平移 O(N^2)
    std::size_t count = 0;
    for (std::size_t i = 0; i < free_list.size(); ++i) {
        const auto region = free_list[i];
        if (count > 0 && free_list[count - 1].offset + free_list[count - 1].bytes == region.offset){
            free_list[count - 1].bytes += region.bytes;
        }
        else{
            free_list[count++] = region;
        }
    }
    free_list.resize(count);
    return {};
}

// first-fit：保留对齐前缀和分配后的余量；没有合适区间时扩展 Arena TODO(可能会产生碎片，我们需不需再freelist当中记录最小和最大，尽量的减少碎片？)
std::expected<std::size_t, std::error_code> ActivationManager::find_or_extend(BackendState& state, std::size_t bytes){
    // 当前采用基础类型对齐；底层 Buffer 基址也必须满足此对齐
    // TODO后端若要求更强的对齐，后续改为从 Backend 查询
    constexpr std::size_t alignment = alignof(std::max_align_t);
    if (bytes == 0)
        return std::size_t{0};

    for (std::size_t i = 0; i < state.free_list.size(); ++i){
        const auto region = state.free_list[i];
        const std::size_t region_end = region.offset + region.bytes;
        if (region_end < region.offset)
            return std::unexpected(std::make_error_code(std::errc::value_too_large));
        // 计算偏移值
        const std::size_t padding = (alignment - region.offset % alignment) % alignment;
        // 计算当前region够不够容纳
        if (padding > region.bytes || bytes > region.bytes - padding)
            continue;

        // ------- 够容纳 --------
        // 当前offset（要从哪里开始）
        const std::size_t offset = region.offset + padding;
        // 当前还剩下多少空间
        const std::size_t remaining = region.bytes - padding - bytes;
        if (padding != 0){
            // push_back 可能扩容，因此不要保留元素引用
            if (remaining != 0)
                state.free_list.push_back({offset + bytes, remaining});
            // 这里一定要新增加一个padding块，否则后续合并的逻辑就有少一块了
            state.free_list[i] = {region.offset, padding};
        }
        else if (remaining != 0){
            // padding==0 ，remaining!=0，说明此时只要修改原来的块即可
            state.free_list[i] = {offset + bytes, remaining};
        }
        else{
            // 如果padding==0 remaining==0 说明此时刚好一个freeList
            state.free_list.erase(state.free_list.begin() + i);
        }
        return offset;
    }

    // 到这里说明没有一个freeList 可以容纳，那就需要在arena追加了
    const std::size_t padding = (alignment - state.arena_bytes % alignment) % alignment;
    const std::size_t offset = state.arena_bytes + padding;
    if (offset < state.arena_bytes)
        return std::unexpected(std::make_error_code(std::errc::value_too_large));
    const std::size_t end = offset + bytes;
    if (end < offset)
        return std::unexpected(std::make_error_code(std::errc::value_too_large));
    if (padding != 0)
        // 这里一定要新增加一个padding块，否则后续合并的逻辑就有少一块了
        state.free_list.push_back({state.arena_bytes, padding});
    state.arena_bytes = end;
    return offset;
}
