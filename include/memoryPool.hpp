#pragma once 


//------- 底层物理分配接口-------
class PhysicalAllocator {
public:
    virtual ~PhysicalAllocator() = default;
    virtual std::expected<void*, std::error_code> raw_alloc(const size_t bytes) = 0;
    virtual void raw_free(void* ptr) = 0;
};
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

// TODO gpu cuda CUDAPhysicalAllocator




// ---- 抽象基类 内存池 ------

class MemoryPool{
protected:
    PhysicalAllocator* _phys_allocator;
public:
    explicit MemoryPool(PhysicalAllocator &allocator)
    : _phys_allocator(allocator) 
    {}

    virtual ~MemoryPool() = default;

    virtual std::expected<void *, std::error_code> alloc(size_t bytes) = 0;

    virtual void free(void *ptr) = 0;
};


