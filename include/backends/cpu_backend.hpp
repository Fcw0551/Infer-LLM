#pragma once 
#include "../../include/backend.hpp"
#include <iostream>
void register_cpu_backend();
// cpu 读取total_ram
size_t get_total_memory();


//-----多态----
class CPUBackend : public Backend
{
public:
    CPUBackend(int device_id)
    :_device_id(device_id)
    {
        std::cout << "cpu 实例化中 device_id: " << device_id << std::endl;
    }

    ~Backend() {
        std::cout<<"cpu 析构中 device_id: " <<device_id<<std::endl;
        for (void* p : _allocations) {
            std::free(p);          
        }
    };

    // 身份（用户 + 调试）
    DeviceType  device_type() const {
        return DeviceType::CPU;
    }
    std::string device_name() const {
        return "CPU:"+std::to_string(_device_id);
    }

    // 是否支持算子，上层schedule调用
    bool supports_op(OperationType op, const Tensor& t) const {
        return true;
    }

    // ========== 3. 内存管理（Scheduler 分配用）==========
    std::expected<void*, std::error_code> alloc(size_t bytes) {
        void* ptr=std::malloc(bytes);
        if(!ptr){
            std::expected(make_error_code(std::errc::no_buffer_space));
        }
        _allocations.push_back(ptr);    
        return ptr;
    }
    void free(void* ptr) {
        if (!ptr) return;
        auto it = std::find(_allocations.begin(), _allocations.end(), ptr);
        if (it != _allocations.end()) _allocations.erase(it);
        return std::free(ptr);
    }

    
    // 跨设备传输  copy结点
    bool is_host() const {
        return true;
    }
    // 从 host 的内存数据 写入这个 tensor 所在的设备的buffer
    std::expected<void, std::error_code> copy_tenosr_set(Tensor *tensor, const void *data, size_t offset, size_t size) ;
    // 从这个 tensor 所在的设备 buffer，读数据到 host 内存。
    std::expected<void, std::error_code> copy_tenosr_get(const Tensor *tensor, void *data, size_t offset, size_t size) ;
    // 设备与设备之间可能要有高速，不需要经过cpu，如果这个方案不行，最后要退回cpu兜底
    //(optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
    bool copy_tensor_high(const Tensor *src, Tensor *dst) const {
        return false;
    }   

    // ========== 5. 执行（Scheduler 分派用）==========
    virtual std::expected<void, std::error_code> compute(Tensor& t) = 0;

    // ========== 6. 同步（Scheduler 跨设备边界用）==========
    virtual void synchronize() {}    // 默认空实现（CPU 用）

    // ========== 7. ★ 预留：BufferPool ==========
    virtual std::expected<BufferPool*, std::error_code>
    create_buffer_pool(BufferKind /*kind*/, size_t /*size*/) {
        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }
    virtual BufferPool* get_buffer_pool(BufferKind /*kind*/) { return nullptr; }
private:
    int _device_id;
    std::vector<void*> _allocations;   // 谁分配的tensor，要由谁释放
};
