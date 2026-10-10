#pragma once 
#include "../../include/backend.hpp"
#include <iostream>
void register_cpu_backend();
// cpu 读取total_ram
size_t get_total_memory();


//-----多态----
class CPUBackend : public Backend{
public:
    CPUBackend(int device_id){
        std::cout<<"cpu 实例化中 device_id"<<device_id<<std::endl;
    }
};