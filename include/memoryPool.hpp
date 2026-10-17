#pragma once 


//------- 底层物理分配接口-------
class PhysicalAllocator {
public:
    virtual ~PhysicalAllocator() = default;
    virtual std::expected<void*, std::error_code> raw_alloc(const size_t bytes) = 0;
    virtual void raw_free(void* ptr) = 0;
};


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