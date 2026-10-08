#pragma once
#include "util.hpp"
#include <string>

//tensor算子的定义
struct Tensor{
    Tensor(const TensorInfo* info)
    :name(info->_name)
    ,offset(info->_offset)
    ,dataType(info->_type)
    ,dims(info->_dims)
    {}
    Tensor(std::string name,DataType type,std::initializer_list<uint64_t> shape)
    :name(name)
    ,dataType(type)
    {
        size_t i=0;
        for(auto&e:shape){
            if(i>=TENSOR_MAX_DIMS){
                break;
            }
            dims[i++]=e;
        }
    }
    std::string name;                                                           //名字
    int layer;                                                                  //第几层
    uint64_t offset;                                                            //数据在内存当中的偏移量
    void* data=nullptr;                                                         //实际指向的数据
    std::array<uint64_t,TENSOR_MAX_DIMS> dims{0,0,0,0};                         //维度
    std::array<uint64_t,TENSOR_MAX_DIMS> strides{0,0,0,0};                      //步长
    std::array<Tensor*,TENSOR_MAX_SRC> src{nullptr,nullptr,nullptr,nullptr};    //某个tensor由哪个tensor计算来
    std::array<uint64_t,OP_PARAMS_MAX_DIMS> op_params{0,0,0,0};                      //算子需要的额外的参数
    enum DataType dataType;                                                     //数据类型
    enum TensorRole tensorRole;                                                 //tensor是作为输入输出还是中间结点
    enum OperationType op;                                                      //算子类型
    enum Device device;                                                         //后端

//---------------------辅助函数-------------------------    
    size_t element_count() const {
        size_t n = 1;
        for (size_t i = 0; i < dims.size(); ++i) {
            if(dims[i]!=0)
            n *= dims[i];
            break;
        }
        return n;
    }

    size_t byte_size() const {
        return element_count() * dataType_size(dataType);
    }

};