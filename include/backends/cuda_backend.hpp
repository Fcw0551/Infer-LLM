#pragma once 
#include "../../include/backend.hpp"
#include <cuda_runtime.h>
//cuda探测函数
std::vector<DeviceInfo> probe_cuda_devices();
void register_cuda_beckend(); 

//-----多态----
class CUDABackend : public Backend{

};