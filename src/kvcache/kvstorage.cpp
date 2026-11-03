#include "../../include/kvcache/kvstorage.hpp"
#include <limits>


void KVStorage::free(StorageId id) noexcept{
    auto it = _states.find(id);
    if (it == _states.end())
        return;
    if (it->second.active_users != 0)
    {
        it->second.pending_free = true;
        return;
    }
    destroy_object(id);
    _states.erase(id);
}


std::expected<KVStorageObject, std::error_code> KVStorage::acquire(StorageId id, std::size_t layer){
    auto it = _states.find(id);
    if (it == _states.end() || it->second.pending_free || it->second.layer != layer || layer >= _backend_layer.size() || !_backend_layer[layer]){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (it->second.active_users == std::numeric_limits<std::size_t>::max()){
        return std::unexpected(std::make_error_code(std::errc::value_too_large));
    }
    // 当前使用者 == 0 
    if (it->second.active_users == 0){
        // 准备数据到指定的layer
        auto prepared = prepare_object(id, layer);
        if (!prepared)
            return std::unexpected(prepared.error());
        it = _states.find(id); // prepare 可能因为新增对象，导致 map rehash
        if (it == _states.end()){
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        }
    }
    auto &state = it->second;
    const auto &object = state.object;
    if (state.pending_free || state.layer != layer || object.id != id || object.backend != _backend_layer[layer] || !object.k_base || !object.v_base){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    ++state.active_users;   // 加加使用者 TODO 临界资源，多人同时写咋办？？？
    return object;
}

void KVStorage::release_access(StorageId id) noexcept{
    auto it = _states.find(id);
    if (it == _states.end() || it->second.active_users == 0)
        return;
    auto &state = it->second;
    --state.active_users;
    if (state.active_users == 0 && state.pending_free){
        destroy_object(id);
        _states.erase(id);
    }
}