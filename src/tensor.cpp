#include "tensor.hpp"
#include "gguf_parser.hpp"

Tensor::Tensor(const TensorInfo* info)
    :name(info->_name)
    ,offset(info->_offset)
    ,dataType(info->_type)
    ,dims(info->_dims)
    {
        size_t i = 0;
        int64_t size = dataType_size(dataType);
        for (auto &e : dims)
        {
            if (i >= TENSOR_MAX_DIMS)
            {
                break;
            }
            dims[i++] = e;
        }
        strides[0] = size;
        strides[1] = size * dims[0];
        strides[2] = size * dims[1] * dims[0];
        strides[3] = size * dims[2] * dims[1] * dims[0];
    }
