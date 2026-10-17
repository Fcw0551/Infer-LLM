#pragma once 
#include "../memoryPool.hpp"

// cpu 

class CPUPhysicalAllocator : public PhysicalAllocator{
public:
    std::expected<void *, std::error_code> raw_alloc(const size_t bytes) override{
        if (bytes == 0){
            return static_cast<void* >(nullptr);
        }

        void *ptr = std::malloc(bytes);

        if (ptr == nullptr){
            return std::unexpected(
                std::make_error_code(std::errc::not_enough_memory));
        }

        return ptr;
    }
    void raw_free(void *ptr) override{
        std::free(ptr);
    }
};

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