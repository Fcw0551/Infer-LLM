#pragma once
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <optional>
#include <expected>
#include <system_error>
#include "util.hpp"
#include "tensor.hpp"
#include<iostream>
#include "graph.hpp"
#include "memoryPool.hpp"

// 探测结果：描述一个可用设备
struct DeviceInfo{
    std::string name; // "cpu" / "cuda:0"
    DeviceType type;
    size_t memory_bytes = 0;
};

// 分配种类：决定进哪个池、生命周期、回收策略
enum class BufferKind : uint8_t{
    WEIGHTS = 0, // 模型级常驻（mmap 导入 / 一次性上传显存）
    ACTIVATION,  // 单次图执行，整块复用
    KVCACHE,     // 请求级预留
};
// 内存池策略
enum class Strategy{
    NAIVE,
    BUMP,
};
// 内存池configh
struct PoolConfig{
    Strategy weights = Strategy::NAVIE;    
    Strategy activation = Strategy::NAIVE; 
    Strategy kvcache = Strategy::NAIVE;
};

class Backend
{
protected:
    int _device_id;                                         // 设备id
    DeviceInfo _info;                                       // 设备信息
    PoolConfig _pool_config;                                // 内存池的config
    std::unique_ptr<PhysicalAllocator> _phys_allocator;     // 实际的，原生的物理接口
    std::unique_ptr<MemoryPool> _weights_pool;              // 权重池
    std::unique_ptr<MemoryPool> _activation_pool;           // 激活池
    std::unique_ptr<MemoryPool> _kvcache_pool;              // kvcache池
public:
    Backend(int device_id,DeviceInfo info,const PoolConfig& cfg)
    :_device_id(device_id)
    ,_info(info)
    ,_pool_configh(cfg)
    {}
    virtual ~Backend() = default;
    // 根据物理原生接口去初始化backend的内存池
    void init_pools(std::unique_ptr<PhysicalAllocator> phys){
        _phys_allocator = std::move(phys);
        _weights_pool = make_pool(_pool_config.weights, *_phys_allocator);
        _activation_pool = make_pool(_pool_config.activation, *_phys_allocator);
        _kvcache_pool = make_pool(_pool_config.kvcache, *_phys_allocator);
    }
    inline std::unique_ptr<MemoryPool> make_pool(PoolConfig::Strategy s, PhysicalAllocator &a){
        switch (s){
        case PoolConfig::Strategy::NAIVE:
            return std::make_unique<NaivePool>(a);
        case PoolConfig::Strategy::BUMP:    // TODO: BumpPool 还没写
        case PoolConfig::Strategy::CACHING: // TODO: CachingPool 还没写
        default:
            return std::make_unique<NaivePool>(a);
        }
    }

    const DeviceInfo& get_info() const { return _info; }
    // 身份（用户 + 调试）
    virtual DeviceType  device_type() const = 0;
    virtual std::string device_name() const = 0;

    // 是否支持算子，上层schedule调用
    virtual bool supports_op(OperationType op, const Tensor& t) const = 0;

    // 内存管理
    // 随意来一块
    virtual std::expected<void *, std::error_code> alloc(size_t bytes) = 0;
    virtual void free(void *ptr) = 0;
    // 选择各种池来一块
    virtual std::expected<void *, std::error_code> alloc(BufferKind kind, size_t bytes) = 0;
    virtual void free(BufferKind kind, void *ptr) = 0;
    
    // 跨设备传输  copy结点
    virtual bool is_host() const = 0;
    // 从 host 的内存数据 写入这个 tensor 所在的设备的buffer
    virtual std::expected<void, std::error_code> copy_tenosr_set(Tensor *tensor, const void *data, size_t offset, size_t size) = 0;
    // 从这个 tensor 所在的设备 buffer，读数据到 host 内存。
    virtual std::expected<void, std::error_code> copy_tenosr_get(const Tensor *tensor, void *data, size_t offset, size_t size) = 0;
    // 设备与设备之间可能要有高速，不需要经过cpu，如果这个方案不行，最后要退回cpu兜底，用的是dst的设备的能力
    //(optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
    virtual bool copy_tensor_high(const Tensor *src, Tensor *dst) const ;   
    
    
    // 执行 scheduler传子图给每个后端执行
    virtual std::expected<void, std::error_code> compute(Graph & subGraph) = 0; 

    // // ==========  同步（Scheduler 跨设备边界用）==========
    // virtual void synchronize() {}    // 默认空实现（CPU 用）

    // // ==========  预留：BufferPool ==========
    // virtual std::expected<BufferPool*, std::error_code> create_buffer_pool(BufferKind /*kind*/, size_t /*size*/) {
    //     return std::unexpected(std::make_error_code(std::errc::not_supported));
    // }
    // virtual BufferPool* get_buffer_pool(BufferKind /*kind*/) { return nullptr; }
};

class BackendRegistry {
public:
    using ProbeFunc = std::function<std::vector<DeviceInfo>()>;
    using CreateFunc = std::function<std::unique_ptr<Backend>(int device_id, const DeviceInfo info, const PoolConfig &cfg)>;
    // 单例
    static BackendRegistry &instance();

    // 注册
    void register_backend(std::string type_name, ProbeFunc probe, CreateFunc creator);

    // 探测
    void probe_all();

    // 创建实例
    std::expected<std::unique_ptr<Backend>, std::error_code> create(std::string_view name, const PoolConfig &cfg) const;

    // 自动选择（优先 CUDA > Metal > CPU，可被环境变量覆盖）
    std::expected<std::unique_ptr<Backend>, std::error_code> create_default(const PoolConfig &cfg) const;

    // 查询探测结果
    const std::vector<DeviceInfo> &available_devices() const;
    std::expected<DeviceInfo,std::error_code> device_info(std::string_view name) const;

    // 辅助解析函数
    static std::string extract_type(std::string_view name);
    static int extract_device_id(std::string_view name);

    //打印函数
    void print(){
        for(auto& e:_devices){
            std::cout << "name:         " << e.name << "\n";
            std::cout << "DeviceType:   " << device_type_name(e.type) << "\n";
            std::cout << "memory_bytes: " << e.memory_bytes << "\n";
        }
    }
private:
    BackendRegistry() = default;
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;
    struct Entry
    {
        ProbeFunc probe;
        CreateFunc create;
    };

    std::unordered_map<std::string, Entry> _entries;    // probe+create
    std::vector<DeviceInfo> _devices;                   // 探测到的设备
};


// 各后端的注册函数 在各自的beckend.hpp/cpp里面
void register_cpu_backend();
void register_cuda_backend();
// 统一注册
void register_all_backends();



