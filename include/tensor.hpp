#pragma once
#include "util.hpp"
#include "gguf_parser.hpp"
#include <string>

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
    int64_t offset;                                                            //数据在内存当中的偏移量
    void* data=nullptr;                                                         //实际指向的数据
    std::array<int64_t,TENSOR_MAX_DIMS> dims{0,0,0,0};                         //维度
    std::array<int64_t,TENSOR_MAX_DIMS> strides{0,0,0,0};                      //步长
    std::array<Tensor*,TENSOR_MAX_SRC> src{nullptr,nullptr,nullptr,nullptr};    //某个tensor由哪个tensor计算来
    int32_t op_params[OP_PARAMS_MAX * sizeof(int32_t)]={0};                         //算子需要的额外的参数 4*4=16字节
    enum DataType dataType;                                                     //数据类型
    enum TensorRole tensorRole;                                                 //tensor是作为输入输出还是中间结点
    enum OperationType op = OperationType::GGML_OP_NONE;                        //算子类型
    enum DeviceType device;                                                     //后端
    Backend* backend=nullptr;                                                   //实际Backend的指针，以方便cuda实现p2p

    //---------------------辅助函数-------------------------
    // 计算元素数量
    size_t element_count() const{
        size_t n = 1;
        for (size_t i = 0; i < dims.size(); ++i)
        {
            if (dims[i] != 0)
                n *= dims[i];
            break;
        }
        return n;
    }
    // 计算tensor大小
    size_t byte_size() const{
        return element_count() * dataType_size(dataType);
    }

};