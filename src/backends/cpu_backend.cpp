#include "../../include/backends/cpu_backend.hpp"
// cpu 读取total_ram
size_t get_total_memory() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}
//cpu 后端的注册函数
void register_cpu_backend(){
    BackendRegistry::instance().register_backend("cpu",
    []()->std::vector<DeviceInfo> { return {DeviceInfo{"cpu", DeviceType::CPU, get_total_memory()}}; },   // probe
    [](int device_id)->std::unique_ptr<Backend> { return std::make_unique<CPUBackend>(device_id); } // create
    );
}


// CPUBackend
// 跨设备传输  copy结点
// 从 host 的内存数据 写入这个 tensor 所在的设备的buffer
std::expected<void, std::error_code> CPUBackend::copy_tenosr_set(Tensor *tensor, const void *data, size_t offset, size_t size) {
    assert(tensor->device==DeviceType::CPU);
    if(data==nullptr){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (offset + bytes > tensor.byte_size()) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    std::memcpy(static_cast<char*>(tensor->data) + offset, data, size);
    return {};
}
// 从这个 tensor 所在的设备 buffer，读数据到 host 内存。
std::expected<void, std::error_code> CPUBackend::copy_tenosr_get(const Tensor *tensor, void *data, size_t offset, size_t size) {
    assert(tensor->device==DeviceType::CPU);

    if (data == nullptr){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (src->data == nullptr) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (offset + bytes > tensor.byte_size()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, bytes);
    return {};
}