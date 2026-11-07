#ifdef INFERLLM_WITH_CUDA

#include "../../include/backends/cuda_backend.hpp"
#include <cstring>

std::vector<DeviceInfo> probe_cuda_devices() {
    std::vector<DeviceInfo> devices;

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess){
        return devices;
    }

    int previous_device = 0;
    const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;

    for (int i = 0; i < count; ++i){
        if (cudaSetDevice(i) != cudaSuccess){
            continue;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;

        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess){
            continue;
        }

        devices.push_back({"cuda:" + std::to_string(i),
                           DeviceType::CUDA,
                           free_bytes});
    }

    if (restore_device){
        cudaSetDevice(previous_device);
    }

    return devices;
}

void register_cuda_backend() {
    BackendRegistry::instance().register_backend(
        "cuda",
        []() -> std::vector<DeviceInfo> {
            return probe_cuda_devices(); // 上面的探测函数
        },
        [](int device_id, DeviceInfo info, const PoolConfig& cfg) -> std::unique_ptr<Backend> {
            return std::make_unique<CUDABackend>(device_id, std::move(info), cfg);
        }
    );
}



//-----cuda backend--------

std::expected<void *, std::error_code> CUDABackend::alloc(BufferKind kind, size_t bytes){
    // 之前探测过，bind 防止操作别的
    // 每次都要bind 子线程不会继承父线程
    cudaSetDevice(_device_id);

    switch (kind){
    case BufferKind::WEIGHTS:
        return _weights_pool->alloc(bytes);
    case BufferKind::ACTIVATION:
        return _activation_pool->alloc(bytes);
    case BufferKind::KVCACHE:
        return _kvcache_pool->alloc(bytes);
    default:
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
}

void CUDABackend::free(BufferKind kind,void* ptr) {
    if (!ptr) return;
    // 之前探测过，bind 防止操作别的
    cudaSetDevice(_device_id);
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

// 设备与设备之间可能要有高速，不需要经过cpu，如果这个方案不行，最后要退回cpu兜底
//(optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
std::expected<bool, std::error_code> CUDABackend::copy_tensor_high(const Tensor *src, Tensor *dst) const{
    //只有cuda不同设备（支持p2p）走NVLink(600GB/s) / PCIe (16-32GB)
    if (!src || !dst || !src->backend || !dst->backend)
        return false;

    auto src_type = src->device;

    // ---以下的源设备都是cuda----
    if (src_type == DeviceType::CUDA){
        auto *src_cuda = dynamic_cast<const CUDABackend* >(src->backend);
        if(src_cuda==nullptr){
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        }

        cudaSetDevice(_device_id);

        if (src_cuda->_device_id == _device_id){
            // 同设备 D2D 同一台设备
            cudaMemcpyAsync(dst->data, src->data, src->byte_size(),cudaMemcpyDeviceToDevice, _stream);
            cudaStreamSynchronize(_stream);
            return true;
        }

        // 跨设备 P2P
        cudaError_t err = cudaMemcpyPeerAsync(dst->data, _device_id, src->data, src_cuda->_device_id, src->byte_size(), _stream);
        if (err != cudaSuccess){
            cudaGetLastError();
            return false;
        }
        cudaStreamSynchronize(_stream);
        return true;
    }

    // 其他后端（Metal、Vulkan 等
    return false;
}



// 执行 scheduler传子图给每个后端执行
std::expected<void, std::error_code> CUDABackend::compute(Graph &graph){
    // 上层scheduler做切分，已经弄好了子图
    // 子图的 nodes 已经是拓扑序——按序遍历
    for (Tensor *node : graph.nodes()){
        // 叶子节点跳过
        if (node->op == OperationType::GGML_OP_NONE){
            continue;
        }

        // 按 op 分派
        auto r = dispatch(node);
        if (!r){
            std::cout << "[cuda] compute failed at node: " << node->name << " op=" << (int)node->op << "\n";
            return r;
        }
    }
    return {};
}



bool CUDABackend::supports_op(OperationType op, const Tensor &t) const{
    // …… ……（TODO）
    return true;
   
}

std::expected<void, std::error_code> CUDABackend::dispatch(Tensor* node) {
    switch (node->op) {
    case OperationType::GGML_OP_ADD:            return check_cuda(CUDA_F32::compute_add(node,_stream));
    case OperationType::GGML_OP_MUL:            return check_cuda(CUDA_F32::compute_mul(node,_stream));
    case OperationType::GGML_OP_UNARY:
        {
            uint32_t op;
            std::memcpy(&op,&node->op_params[0],4);
            
            switch(static_cast<OperationType_unary_op>(op)){
            case OperationType_unary_op::GGML_UNARY_OP_SILU:
                return check_cuda(CUDA_F32::compute_silu(node,_stream));
            default:
                return std::unexpected(std::make_error_code(std::errc::not_supported));
            }
           
        }
    case OperationType::GGML_OP_RMS_NORM:       return check_cuda(CUDA_F32::compute_rms_norm(node,_stream));
    case OperationType::GGML_OP_GET_ROWS:       return check_cuda(CUDA_F32::compute_get_rows(node,_stream));
    case OperationType::GGML_OP_MUL_MAT:        return check_cuda(CUDA_F32::compute_mul_mat(node,_stream));
    case OperationType::GGML_OP_RESHAPE:        return check_cuda(CUDA_F32::compute_reshape(node,_stream));
    case OperationType::GGML_OP_ROPE:           return check_cuda(CUDA_F32::compute_rope(node,_stream));
    case OperationType::GGML_OP_FLASH_ATTN_EXT: return check_cuda(CUDA_F32::compute_flash_attn(node,_stream));
    case OperationType::GGML_OP_CPY:            return check_cuda(CUDA_F32::compute_cpy(node,_stream));
    case OperationType::GGML_OP_VIEW:           return check_cuda(CUDA_F32::compute_view(node,_stream));

    case OperationType::GGML_OP_NONE:
        return {};   // 叶子，跳过

    default:
        std::cerr << "[cuda] unsupported op: " << (int)node->op << "\n";
        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }
}


#endif // INFERLLM_WITH_CUDA
