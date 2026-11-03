#include "../../include/kvcache/kvcacheManager.hpp"

// 初始化缓存容量、各层计算设备、相关配置信息
std::expected<void, std::error_code> KVCacheManager::init(std::span<const KVLayerSpec> layers, std::span<Backend *const> backend_layer, const KVCacheConfig &config){
    return _strategy->init(layers, backend_layer, config);
}

// 创建一条新序列，返回其身份编号
std::expected<SequenceId, std::error_code> KVCacheManager::create_sequence(){
    SequenceId s_id = _next_sequence_id++;
    // 先登记本层状态，避免底层创建成功后，本层登记再因分配失败而留下孤立序列。
    auto [it, inserted] = _sequences.try_emplace(s_id);
    if (!inserted){
        return std::unexpected(std::make_error_code(std::errc::file_exists));
    }
    try {
        auto result = _strategy->create_sequence(s_id);
        if(!result){
            _sequences.erase(s_id);
            return std::unexpected(result.error());
        }
    } catch (...) {
        _sequences.erase(s_id);
        throw;
    }
    return s_id;  
}

// 为某条序列预留 n_tokens 个新增 token 的缓存空间
// range = { committed_length, n_tokens }；已有 pending 时返回
// std::errc::device_or_resource_busy；容量不足返回 std::errc::not_enough_memory
std::expected<void, std::error_code> KVCacheManager::reserve_append(SequenceId id, std::size_t n_tokens){
    if(!_sequences.contains(id)){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    auto &state = _sequences.at(id);
    if (state.pending){
        return std::unexpected(std::make_error_code(std::errc::device_or_resource_busy));
    }
    // 计算预留位置
    TokenRange t={.begin = state.committed_length, .count = n_tokens};
    auto result = _strategy->reserve_append(id, t);
    if (!result){
        return std::unexpected(result.error());
    }
    state.pending=t;
    return {};
}

// 获取某层的 K/V 视图，包含本轮 pending 范围
// 要求该层已 reserve_append，且 pending 非空
// TODO 多次申请同一块，临界资源问题
std::expected<KVLayerAccess, std::error_code> KVCacheManager::acquire_layer(SequenceId id, std::size_t layer){
    // 获取之前肯定有预留，然后外层就在预留的空间上追加即可
    if(!_sequences.contains(id)){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    auto &state = _sequences.at(id);
    if(!state.pending){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    return _strategy->acquire_layer(id, layer, *state.pending);
}

std::expected<void, std::error_code> KVCacheManager::release_layer(KVLayerAccess& access) noexcept {
    return _strategy->release_layer(access);
}

// 所有层计算成功后提交本轮追加：committed_length += pending.count
std::expected<void, std::error_code> KVCacheManager::commit_append(SequenceId id){
    if (!_sequences.contains(id)){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    auto &state = _sequences.at(id);
    if (!state.pending){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    const TokenRange range = *state.pending;

    // commit_append
    auto result = _strategy->commit_append(id, range);
    if (!result){
        return std::unexpected(result.error());
    }

    state.committed_length = range.end();
    state.pending.reset();

    return {};
}

// 计算失败时取消本轮追加，保留原有有效缓存
std::expected<void, std::error_code> KVCacheManager::abort_append(SequenceId id){
    if (!_sequences.contains(id)){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    auto &state = _sequences.at(id);
    if (!state.pending){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    const TokenRange range = *state.pending;
    auto result = _strategy->abort_append(id, range);
    if (!result){
        return std::unexpected(result.error());
    }

    state.pending.reset(); // committed_length 不变

    return {};
}

// 查询已提交的 token 数量（不含 pending）
std::expected<std::size_t, std::error_code> KVCacheManager::length(SequenceId id) const{
    if(!_sequences.contains(id)){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    return _sequences.at(id).committed_length;
}

// 释放序列持有的全部缓存引用，并注销 ID
std::expected<void, std::error_code> KVCacheManager::release_sequence(SequenceId id){
    if(!_sequences.contains(id)){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    auto result = _strategy->release_sequence(id);
    if(!result){
        return std::unexpected(result.error());
    }
    _sequences.erase(id);           // 删掉id
    return {};
}

// 清空全部序列 底层容量保留，供下次使用
std::expected<void, std::error_code> KVCacheManager::clear_sequence(){
    auto result = _strategy->clear_sequence();
    if(!result){
        return std::unexpected(result.error());
    }
    _sequences.clear();
    return {};
}
