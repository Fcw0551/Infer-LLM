#pragma once
#include "util.h"
#include <string>

//tensor算子的定义
struct Tensor{
    std::string name;                                   //名字
    int layer;                                          //第几层
    uint64_t offset;                                    //数据在内存当中的偏移量
    void* data=nullptr;                                 //实际指向的数据
    std::array<uint64_t,TENSOR_MAX_DIMS> dims{0,0,0,0};   //维度
    std::array<uint64_t,TENSOR_MAX_DIMS> strides{0,0,0,0};//步长
    std::array<Tensor*,TENSOR_MAX_SRC> src{nullptr,nullptr,nullptr,nullptr};    //某个tensor由哪个tensor计算来
    enum DateType dataType                              //数据类型
    enum TensorRole tensorRole;                         //tensor是作为输入输出还是中间结点
    enum OperationType op;                              //算子类型
    enum Device device                                  //后端

//---------------------辅助函数-------------------------    
    size_t element_count() const {
        size_t n = 1;
        for (int i = 0; i < n_dims; ++i) n *= ne[i];
        return n;
    }

    size_t byte_size() const {
        return element_count() * dtype_size(type);
    }

    bool is_leaf() const { return op == Op::NONE; }
}