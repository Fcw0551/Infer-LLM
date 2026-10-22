#pragma once 
#include "../memoryPool.hpp"


// 每次调用原始的api进行分配，没有任何的策略
class NaivePool : public MemoryPool{
public:
    explicit NaivePool(PhysicalAllocator& allocator)
    :MemoryPool(allocator) 
    {}
    std::expected<void *, std::error_code> alloc(const size_t bytes){
        auto result = _phys_allocator.raw_alloc(bytes);
        if(!result){
            return unexpected(result.error());
        }
        return result;
    }
    void free(void* ptr){
        _phys_allocator.raw_free(ptr);
    }
};