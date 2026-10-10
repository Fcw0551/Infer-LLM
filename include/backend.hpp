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

// 探测结果：描述一个可用设备
struct DeviceInfo{
    std::string name; // "cpu" / "cuda:0"
    DeviceType type;
    size_t memory_bytes = 0;
};

class Backend
{
public:
//     virtual ~Backend() = default;

//     // ---- 身份 ----
//     virtual DeviceType device_type() const = 0;
//     virtual std::string device_name() const = 0;

//     // ---- 能力查询 ----
//     virtual bool supports_op(OperationType op, const Tensor &t) const = 0;

//     // ---- 内存（底层裸分配）----
//     virtual std::expected<void *, std::error_code> alloc(size_t bytes) = 0;
//     virtual void free(void *ptr) = 0;

//     // ---- 跨设备拷贝 ----
//     virtual std::expected<void, std::error_code>
//     copy_to(Backend &dst_backend, void *dst, const void *src, size_t bytes) = 0;

//     // ---- 执行 ----
//     virtual std::expected<void, std::error_code> compute(Tensor &t) = 0;

//     // ---- 同步（CUDA stream 等）----
//     virtual void synchronize() {}

//     // ---- 探测（静态方法，问"这台机器上有几个这种设备"）----
//     virtual std::vector<DeviceInfo> probe() const = 0;

//     // ---- BufferPool ----
//     //virtual std::expected<BufferPool *, std::error_code>
//     // create_buffer_pool(BufferKind /*kind*/, size_t /*initial_size*/)
//     // {
//     //     return std::unexpected(std::make_error_code(std::errc::not_supported));
//     // }
//    // virtual BufferPool *get_buffer_pool(BufferKind /*kind*/) { return nullptr; }
};

class BackendRegistry {
public:
    using ProbeFunc = std::function<std::vector<DeviceInfo>()>;
    using CreateFunc = std::function<std::unique_ptr<Backend>(int device_id)>;
    // 单例
    static BackendRegistry &instance();

    // 注册
    void register_backend(std::string type_name, ProbeFunc probe, CreateFunc creator);

    // 探测
    void probe_all();

    // 创建实例
    std::expected<std::unique_ptr<Backend>, std::error_code> create(std::string_view name) const;

    std::expected<std::unique_ptr<Backend>, std::error_code> create_default() const;

    // 查询探测结果
    const std::vector<DeviceInfo> &available_devices() const;
    std::expected<DeviceInfo,std::error_code> device_info(std::string_view name) const;


    // 自动选择（优先 CUDA > Metal > CPU，可被环境变量覆盖）

    // 辅助解析函数
    static std::string extract_type(std::string_view name);
    static int extract_device_id(std::string_view name);

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



