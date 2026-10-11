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



//-----cuda backend--------



CUDABackend::alloc(size_t bytes) {
    // 之前探测过，bind 防止操作别的
    // 每次都要bind 子线程不会继承父线程
    cudaSetDevice(_device_id);

    void* p = nullptr;
    cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) {
        return std::unexpected(std::make_error_code(std::errc::not_enough_memory));
    }
    _allocations.push_back(p);
    return p;
}

void CUDABackend::free(void* ptr) {
    if (!ptr) return;
    // 之前探测过，bind 防止操作别的
    cudaSetDevice(_device_id);
    cudaFree(ptr);

    auto it = std::find(_allocations.begin(), _allocations.end(), ptr);
    if (it != _allocations.end()) _allocations.erase(it);
}

 // 从 host 的内存数据 写入这个 tensor 所在的设备的buffer
std::expected<void, std::error_code> CUDABackend::copy_tensor_set(Tensor* tensor, const void* data, size_t offset, size_t size) {
    assert(tensor->device==DeviceType::CUDA);
    if (!tensor || !tensor->data || !data) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (offset + size > tensor->byte_size()) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    cudaSetDevice(_device_id);
    // host -> device
    char* dst_ptr = static_cast<char*>(tensor->data) + offset;
    cudaError_t err = cudaMemcpyAsync(dst_ptr, data, size, cudaMemcpyHostToDevice, _stream);
    if (err != cudaSuccess) {
        return std::unexpected(std::make_error_code(std::errc::io_error));
    }

    // 同步（先简单处理，将来可以异步）(TODO)
    cudaStreamSynchronize(_stream);
    return {};
}
// 从这个 tensor 所在的设备 buffer，读数据到 host 内存。
std::expected<void, std::error_code> CUDABackend::copy_tensor_get(const Tensor* tensor, void* data, size_t offset, size_t size) {
    assert(tensor->device==DeviceType::CUDA);
    
    if (!tensor || !tensor->data || !data) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (offset + size > tensor->byte_size()) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    cudaSetDevice(_device_id);

    // device → host
    const char* src_ptr = static_cast<const char*>(tensor->data) + offset;
    cudaError_t err = cudaMemcpyAsync(data, src_ptr, size,cudaMemcpyDeviceToHost,_stream);
    if (err != cudaSuccess) {
        return std::unexpected(std::make_error_code(std::errc::io_error));
    }
    //(TODO)
    cudaStreamSynchronize(_stream);
    return {};
}







#endif // INFERLLM_WITH_CUDA