#ifdef INFERLLM_WITH_CUDA

#include "../../include/backends/cuda_backend.hpp"

std::vector<DeviceInfo> probe_cuda_devices() {
    std::vector<DeviceInfo> devices;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
        return devices;
    for (int i = 0; i < count; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        devices.push_back({"cuda:" + std::to_string(i),
                           DeviceType::CUDA,
                           prop.totalGlobalMem});
    }
    return devices;
}

void register_cuda_backend() {
    BackendRegistry::instance().register_backend(
        "cuda",
        []() -> std::vector<DeviceInfo> {
            return probe_cuda_devices(); // 上面的探测函数
        },
        [](int device_id) -> std::unique_ptr<Backend> {
            return std::make_unique<CUDABackend>(device_id);
        }
    );
}

#endif // INFERLLM_WITH_CUDA