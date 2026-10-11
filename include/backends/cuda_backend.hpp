#pragma once 
#include "../../include/backend.hpp"
#include <cuda_runtime.h>
//cuda探测函数
std::vector<DeviceInfo> probe_cuda_devices();
void register_cuda_backend(); 

//-----多态----
// TODO（优化的点还有很多，现在是实现一个基础版本）
class CUDABackend : public Backend{
public:
   CUDABackend(int device_id) exp
    :_device_id(device_id)
    {
        std::cout << "cuda 实例化中 device_id: " << device_id << std::endl;
        // 切换到目标设备
        cudaSetDevice(_device_id);

        //  创建 stream
        cudaStream_t stream;
        cudaError_t err = cudaStreamCreate(&stream);                //(TODO，只创建了一条)
        if (err != cudaSuccess){
            throw std::runtime_error("cudaStreamCreate failed");    //先用异常，我们的推理框架粗心是不用异常体系的，
        }
        _stream = stream;
    }

    ~Backend() {
        std::cout<<"cuda 析构中 device_id: " <<device_id<<std::endl;
        // 先释放所有分配
        for (void *p : _allocations)
            cudaFree(p);
        _allocations.clear();

        // 销毁 stream
        if (_stream){
            cudaSetDevice(_device_id);
            cudaStreamDestroy(static_cast<cudaStream_t>(_stream));
            _stream = nullptr;
        }
    };

    // 身份（用户 + 调试）
    DeviceType  device_type() const {
        return DeviceType::CPU;
    }
    std::string device_name() const {
        return "CUDA:"+std::to_string(_device_id);
    }

    // 是否支持算子，上层schedule调用
    bool supports_op(OperationType op, const Tensor& t) const {
        switch (op){
            
        //…… ……（TODO）

        default:
            return false;
        }
    }

    // 内存管理
    std::expected<void*, std::error_code> alloc(size_t bytes) = 0;
    void free(void* ptr)   = 0;

    // 跨设备传输  copy结点
    bool is_host() const {
        return false;
    }
    // 从 host 的内存数据 写入这个 tensor 所在的设备的buffer
    std::expected<void, std::error_code> copy_tenosr_set(Tensor *tensor, const void *data, size_t offset, size_t size);
    // 从这个 tensor 所在的设备 buffer，读数据到 host 内存。
    std::expected<void, std::error_code> copy_tenosr_get(Tensor *tensor, void *data, size_t offset, size_t size);
    // 设备与设备之间可能要有高速，不需要经过cpu，如果这个方案不行，最后要退回cpu兜底
    //(optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
    bool copy_tensor_high(const Tensor *src, Tensor *dst) const ;   
                                                   
                                                        

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
    cudaStream_t _stream;
};