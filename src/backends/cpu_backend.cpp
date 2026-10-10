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