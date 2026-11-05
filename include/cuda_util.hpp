#pragma once 
#include <cuda_runtime_api.h>
#include <expected>
#include <system_error>
#include <iostream>
//检查cuda的返回值
inline std::expected<void, std::error_code> check_cuda(cudaError_t error){
    if (error == cudaSuccess){
        return {};
    }

    std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';

    // 暂时映射为通用错误；后续可用自定义 error_category 保留 CUDA 错误码
    return std::unexpected(std::make_error_code(std::errc::io_error));
}
