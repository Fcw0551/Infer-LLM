#pragma once
#include "util.hpp"
#include "gguf_parser.hpp"
#include <string>
#include <iostream>
class Backend;  //前置声明

//tensor算子的定义
struct Tensor{
    Tensor(const TensorInfo* info)
    :name(info->_name)
    ,offset(info->_offset)
    ,dataType(info->_type)
    ,dims(info->_dims)
    {}
    Tensor(std::string_view name,DataType type,std::initializer_list<int64_t> shape)
    :name(std::string(name))
    ,dataType(type)
    {
        size_t i=0;
        int64_t size=dataType_size(type);
        for(auto&e:shape){
            if(i>=TENSOR_MAX_DIMS){
                break;
            }
            dims[i++]=e;
        }
        strides[0]=size;
        strides[1]=size*dims[0];
        strides[2]=size*dims[1]*dims[0];
        strides[3]=size*dims[2]*dims[1]*dims[0];

    }
    std::string name;                                                           //名字
    int layer;                                                                  //第几层
    int64_t offset;                                                             //数据在内存当中的偏移量（相对于gguf的数据区）
    void* data=nullptr;                                                         //实际指向的数据
    std::array<int64_t,TENSOR_MAX_DIMS> dims{1,1,1,1};                          //维度
    std::array<int64_t,TENSOR_MAX_DIMS> strides{0,0,0,0};                       //步长
    std::array<Tensor*,TENSOR_MAX_SRC> src{nullptr,nullptr,nullptr,nullptr};    //某个tensor由哪个tensor计算来
    int32_t op_params[OP_PARAMS_MAX * sizeof(int32_t)]={0};                     //算子需要的额外的参数 4*4=16字节
    enum DataType dataType=DataType::GGML_TYPE_F32;                             //数据类型
    enum TensorRole tensorRole=TensorRole::TENSOR_ROLE_NONE;                    //tensor是作为输入输出还是中间结点
    enum OperationType op = OperationType::GGML_OP_NONE;                        //算子类型
    enum DeviceType device=DeviceType::CPU;                                     //后端
    Backend* backend=nullptr;                                                   //实际Backend的指针
    bool pinned = false;                                                        //当前算子是否被pinned在某个backend上
    //---------------------辅助函数-------------------------
    // 计算元素数量
    size_t element_count() const{
        return dims[0] * dims[1] * dims[2] * dims[3];
    }
    // 计算tensor大小
    size_t byte_size() const{
        const size_t n = element_count();
        switch (dataType){
        // 32 个元素一块
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_IQ4_NL:
            return (n / 32) * dataType_size(dataType);
        // 256 个元素一块
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
            return (n / 256) * dataType_size(dataType);
        default:
            return n * dataType_size(dataType);
        }
    }
    void print()
    {
        std::cerr << "name: " << name
                  << "elements: " << element_count()
                  << "dims"<<dims[0]<<" "<<dims[1]<<" "<<dims[2]<<" "<<dims[3]
                  <<std::endl;
    }
};