#pragma once 
#include "../../include/backend.hpp"
#include <iostream>
#include <thread>
#include <mutex>
#include<condition_variable>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
// 线程池
class ThreadPool
{
public:
    ThreadPool() = default;
    ~ThreadPool() { shutdown(); }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    void init(int n_threads);
    void shutdown();

    int size() const { return _n_threads; }

    // 把 [0, n) 的任务并行分给所有线程
    void parallel_for(int64_t n, const std::function<void(int64_t)> &fn);

private:
    void worker_loop(int tid);

    std::vector<std::thread> _threads;
    int _n_threads = 0;

    // 任务状态
    std::function<void(int64_t)> _fn; // 当前任务
    int64_t _n = 0;                   // 任务总数
    std::atomic<int64_t> _next{0};    // 下一个要处理的下标

    // 同步
    std::mutex _mtx;
    std::condition_variable _cv_start; // 唤醒 worker 开始干活
    std::condition_variable _cv_done;  // 通知主线程干完了
    int _n_done = 0;                   // 完成的 worker 数
    int _epoch = 0;                    // 任务轮次（防止伪唤醒）
    bool _stop = false;                // 停止标志
};


// --------- cpu 注册模块---------------------
void register_cpu_backend();
// cpu 读取total_ram
size_t get_total_memory();


//-----多态----

class CPUBackend : public Backend
{
public:
    CPUBackend(int device_id,DeviceInfo info,const PoolConfig&cfg)
    :Backend(device_id,info,cfg)
    {
        init_pools(std::make_unique<CPUPhysicalAllocator>());
        std::cout << "cpu 实例化中 device_id: " << _device_id << std::endl;
        cpu_init();
    }

    ~CPUBackend() {
        std::cout<<"cpu 析构中 device_id: " <<_device_id<<std::endl;
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
    

    // 设置线程数量
    void set_n_threads(size_t n){
        _n_threads=n;
        _threadPool.init(n);   //重建线程池，刚刚cpu_init建过
    }


    // 内存管理
    std::expected<void* , std::error_code> alloc(BufferKind kind, size_t bytes) override{
        switch (kind)
        {
        case BufferKind::WEIGHTS:
            return _weights_pool->alloc(bytes);
        case BufferKind::ACTIVATION:
            return _activation_pool->alloc(bytes);
        case BufferKind::KVCACHE:
            return _kvcache_pool->alloc(bytes);
        default:
            return std::unexpected(
                std::make_error_code(std::errc::invalid_argument));
        }
    }
    void free(BufferKind kind, void *ptr) override{
        switch (kind){
        case BufferKind::WEIGHTS:
            _weights_pool->free(ptr);
            break;
        case BufferKind::ACTIVATION:
            _activation_pool->free(ptr);
            break;
        case BufferKind::KVCACHE:
            _kvcache_pool->free(ptr);
            break;
        default:
            break;
        }
    }

    // 跨设备传输  copy结点
    bool is_host() const {
        return true;
    }
    // 从 host 的内存数据 写入这个 tensor 所在的设备的buffer
    std::expected<void, std::error_code> copy_tensor_set(Tensor *tensor, const void *data, size_t offset, size_t size) ;
    // 从这个 tensor 所在的设备 buffer，读数据到 host 内存。
    std::expected<void, std::error_code> copy_tensor_get(const Tensor *tensor, void *data, size_t offset, size_t size) ;
    // 设备与设备之间可能要有高速，不需要经过cpu，如果这个方案不行，最后要退回cpu兜底
    //(optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
    std::expected<bool, std::error_code> copy_tensor_high(const Tensor *src, Tensor *dst) const {
        return false;
    }   

    // 执行 scheduler传子图给每个后端执行
    virtual std::expected<void, std::error_code> compute(Graph & subGraph) ;
  
    std::expected<void, std::error_code> dispatch(Tensor* node);
    
    //  同步（Scheduler 跨设备边界用）
    void synchronize() {}    // 默认空实现（CPU 用）

    // ========== 7. ★ 预留：BufferPool ==========
    // virtual std::expected<BufferPool*, std::error_code>
    // create_buffer_pool(BufferKind /*kind*/, size_t /*size*/) {
    //     return std::unexpected(std::make_error_code(std::errc::not_supported));
    // }
    // virtual BufferPool* get_buffer_pool(BufferKind /*kind*/) { return nullptr; }

private:
    // cpu初始化
    void cpu_init();

    // ---------算子实现----------    
    std::expected<void, std::error_code> compute_get_rows(Tensor* dst);
    std::expected<void, std::error_code> compute_rms_norm(Tensor* dst);
    std::expected<void, std::error_code> compute_mul_mat(Tensor* dst);
    std::expected<void, std::error_code> compute_rope(Tensor* dst);
    std::expected<void, std::error_code> compute_flash_attn(Tensor* dst);
    std::expected<void, std::error_code> compute_add(Tensor* dst);
    std::expected<void, std::error_code> compute_silu(Tensor* dst);
    std::expected<void, std::error_code> compute_mul(Tensor* dst);


    // 辅助检查函数
    void infer_are_same_shape(const Tensor* x,const Tensor* y){
        assert(x->dims[0]==y->dims[0]);
        assert(x->dims[1]==y->dims[1]);
        assert(x->dims[2]==y->dims[2]);
        assert(x->dims[3]==y->dims[3]);
    }
private:
    //  int _device_id;
    //  DeviceInfo _info;
    //  std::unique_ptr<MemoryPool> _weights_pool;      // 权重池
    //  std::unique_ptr<MemoryPool> _activation_pool;   // 激活池
    //  std::unique_ptr<MemoryPool> _kvcache_pool;      // kvcache池
    int _n_threads=1;                  // 线程数
    ThreadPool _threadPool;            // 线程池
};
