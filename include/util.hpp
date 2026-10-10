#pragma once 
#include <cstddef>
#include <cstdint>
#define GGUF_MAX_DIMS 4
#define TENSOR_MAX_DIMS 4
#define TENSOR_MAX_SRC 4
#define OP_PARAMS_MAX 4
// types that can be stored as GGUF KV data
enum GGUFType {
        GGUF_TYPE_UINT8   = 0,
        GGUF_TYPE_INT8    = 1,
        GGUF_TYPE_UINT16  = 2,
        GGUF_TYPE_INT16   = 3,
        GGUF_TYPE_UINT32  = 4,
        GGUF_TYPE_INT32   = 5,
        GGUF_TYPE_FLOAT32 = 6,
        GGUF_TYPE_BOOL    = 7,
        GGUF_TYPE_STRING  = 8,
        GGUF_TYPE_ARRAY   = 9,
        GGUF_TYPE_UINT64  = 10,
        GGUF_TYPE_INT64   = 11,
        GGUF_TYPE_FLOAT64 = 12,
        GGUF_TYPE_COUNT,       // marks the end of the enum
};

//types that can be stored as GGUF tensor type
enum DataType {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_I16     = 25,
        GGML_TYPE_I32     = 26,
        GGML_TYPE_I64     = 27,
        GGML_TYPE_F64     = 28,
        GGML_TYPE_IQ1_M   = 29,
        GGML_TYPE_BF16    = 30,
        // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        // GGML_TYPE_Q4_0_4_8 = 32,
        // GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
        // GGML_TYPE_IQ4_NL_4_4 = 36,
        // GGML_TYPE_IQ4_NL_4_8 = 37,
        // GGML_TYPE_IQ4_NL_8_8 = 38,
        GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
        GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
        GGML_TYPE_Q1_0    = 41,
        GGML_TYPE_Q2_0    = 42,
        GGML_TYPE_COUNT   = 43, //作为一个标识的
};

enum class TensorRole:uint8_t{
    TENSOR_ROLE_NONE       = 0,
    TNESOR_ROLE_INPUT      = 1 << 0,
    TENSOR_ROLE_OUTPUT     = 1 << 1,    
    TENSOR_ROLE_WEIGHT     = 1 << 2,    //权重
    TENSOR_ROLE_ACTIVATION = 1 << 3,    //计算结果中间产生的
    TENSOR_ROLE_VIEW       = 1 << 4,    //视图
};

//算子类型
enum class OperationType:uint32_t {
        GGML_OP_NONE = 0,

        GGML_OP_DUP,
        GGML_OP_ADD,
        GGML_OP_ADD_ID,
        GGML_OP_ADD1,
        GGML_OP_ACC, 
        GGML_OP_SUB,
        GGML_OP_MUL,
        GGML_OP_DIV,
        GGML_OP_SQR,
        GGML_OP_SQRT,
        GGML_OP_LOG,
        GGML_OP_SIN,
        GGML_OP_COS,
        GGML_OP_SUM,
        GGML_OP_SUM_ROWS,
        GGML_OP_CUMSUM,
        GGML_OP_MEAN,
        GGML_OP_ARGMAX,
        GGML_OP_COUNT_EQUAL,
        GGML_OP_REPEAT,
        GGML_OP_REPEAT_BACK,
        GGML_OP_CONCAT,
        GGML_OP_SILU_BACK,
        GGML_OP_NORM, // normalize
        GGML_OP_RMS_NORM,
        GGML_OP_RMS_NORM_BACK,
        GGML_OP_GROUP_NORM,
        GGML_OP_L2_NORM,

        GGML_OP_MUL_MAT,
        GGML_OP_MUL_MAT_ID,
        GGML_OP_OUT_PROD,

        GGML_OP_SCALE,
        GGML_OP_SET,
        GGML_OP_CPY,
        GGML_OP_CONT,
        GGML_OP_RESHAPE,
        GGML_OP_VIEW,
        GGML_OP_PERMUTE,
        GGML_OP_TRANSPOSE,
        GGML_OP_GET_ROWS,
        GGML_OP_GET_ROWS_BACK,
        GGML_OP_SET_ROWS,
        GGML_OP_DIAG,
        GGML_OP_DIAG_MASK_INF,
        GGML_OP_DIAG_MASK_ZERO,
        GGML_OP_SOFT_MAX,
        GGML_OP_SOFT_MAX_BACK,
        GGML_OP_ROPE,
        GGML_OP_ROPE_BACK,
        GGML_OP_CLAMP,
        GGML_OP_CONV_TRANSPOSE_1D,
        GGML_OP_IM2COL,
        GGML_OP_IM2COL_BACK,
        GGML_OP_IM2COL_3D,
        GGML_OP_COL2IM_1D,
        GGML_OP_CONV_2D,
        GGML_OP_CONV_3D,
        GGML_OP_CONV_2D_DW,
        GGML_OP_CONV_TRANSPOSE_2D,
        GGML_OP_POOL_1D,
        GGML_OP_POOL_2D,
        GGML_OP_POOL_2D_BACK,
        GGML_OP_UPSCALE,
        GGML_OP_PAD,
        GGML_OP_PAD_REFLECT_1D,
        GGML_OP_ROLL,
        GGML_OP_ARANGE,
        GGML_OP_TIMESTEP_EMBEDDING,
        GGML_OP_ARGSORT,
        GGML_OP_TOP_K,
        GGML_OP_LEAKY_RELU,
        GGML_OP_TRI,
        GGML_OP_FILL,

        GGML_OP_FLASH_ATTN_EXT,
        GGML_OP_FLASH_ATTN_BACK,
        GGML_OP_SSM_CONV,
        GGML_OP_SSM_SCAN,
        GGML_OP_WIN_PART,
        GGML_OP_WIN_UNPART,
        GGML_OP_GET_REL_POS,
        GGML_OP_ADD_REL_POS,
        GGML_OP_RWKV_WKV6,
        GGML_OP_GATED_LINEAR_ATTN,
        GGML_OP_RWKV_WKV7,
        GGML_OP_SOLVE_TRI,
        GGML_OP_GATED_DELTA_NET,
        GGML_OP_LIGHTNING_INDEXER,
        GGML_OP_DSV4_HC_COMB,
        GGML_OP_DSV4_HC_PRE,
        GGML_OP_DSV4_HC_POST,

        GGML_OP_UNARY,

        GGML_OP_MAP_CUSTOM1,
        GGML_OP_MAP_CUSTOM2,
        GGML_OP_MAP_CUSTOM3,

        GGML_OP_CUSTOM,

        GGML_OP_CROSS_ENTROPY_LOSS,
        GGML_OP_CROSS_ENTROPY_LOSS_BACK,
        GGML_OP_OPT_STEP_ADAMW,
        GGML_OP_OPT_STEP_SGD,

        GGML_OP_GLU,

        GGML_OP_COUNT,
};
 enum class OperationType_unary_op: uint32_t {
        GGML_UNARY_OP_ABS,
        GGML_UNARY_OP_SGN,
        GGML_UNARY_OP_NEG,
        GGML_UNARY_OP_STEP,
        GGML_UNARY_OP_TANH,
        GGML_UNARY_OP_ELU,
        GGML_UNARY_OP_RELU,
        GGML_UNARY_OP_SIGMOID,
        GGML_UNARY_OP_GELU,
        GGML_UNARY_OP_GELU_QUICK,
        GGML_UNARY_OP_SILU,
        GGML_UNARY_OP_HARDSWISH,
        GGML_UNARY_OP_HARDSIGMOID,
        GGML_UNARY_OP_EXP,
        GGML_UNARY_OP_EXPM1,
        GGML_UNARY_OP_SOFTPLUS,
        GGML_UNARY_OP_GELU_ERF,
        GGML_UNARY_OP_XIELU,
        GGML_UNARY_OP_FLOOR,
        GGML_UNARY_OP_CEIL,
        GGML_UNARY_OP_ROUND,
        GGML_UNARY_OP_TRUNC,

        GGML_UNARY_OP_COUNT,
};


enum class DeviceType:uint8_t{
    CPU=0,
    CUDA=1,
    SYCL=2,
    VULKAN=3,
    METAL=4,
};



// ============ 枚举转字符串 ============
const char* gguf_value_type_name(GGUFType t);

const char* ggml_type_name(DataType t);


size_t dataType_size(DataType dataType);