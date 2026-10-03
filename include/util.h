#pragma once 

#define GGUF_MAX_DIMS 4


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
enum TensorType {
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

// ============ 枚举转字符串 ============
// 注意：GGUFType / TensorType 是非作用域枚举，枚举量在全局作用域（GGUF_TYPE_* / GGML_TYPE_*），
// 不是 GGUFType::UINT8 这种写法。头文件里的定义要 inline，否则被多个 TU 包含会重复定义。
inline const char* gguf_value_type_name(GGUFType t) {
    switch (t) {
    case GGUF_TYPE_UINT8:   return "UINT8";
    case GGUF_TYPE_INT8:    return "INT8";
    case GGUF_TYPE_UINT16:  return "UINT16";
    case GGUF_TYPE_INT16:   return "INT16";
    case GGUF_TYPE_UINT32:  return "UINT32";
    case GGUF_TYPE_INT32:   return "INT32";
    case GGUF_TYPE_FLOAT32: return "FLOAT32";
    case GGUF_TYPE_BOOL:    return "BOOL";
    case GGUF_TYPE_STRING:  return "STRING";
    case GGUF_TYPE_ARRAY:   return "ARRAY";
    case GGUF_TYPE_UINT64:  return "UINT64";
    case GGUF_TYPE_INT64:   return "INT64";
    case GGUF_TYPE_FLOAT64: return "FLOAT64";
    default:                return "UNKNOWN";
    }
}

inline const char* ggml_type_name(TensorType t) {
    switch (t) {
    case GGML_TYPE_F32:     return "F32";
    case GGML_TYPE_F16:     return "F16";
    case GGML_TYPE_Q4_0:    return "Q4_0";
    case GGML_TYPE_Q4_1:    return "Q4_1";
    case GGML_TYPE_Q5_0:    return "Q5_0";
    case GGML_TYPE_Q5_1:    return "Q5_1";
    case GGML_TYPE_Q8_0:    return "Q8_0";
    case GGML_TYPE_Q8_1:    return "Q8_1";
    case GGML_TYPE_Q2_K:    return "Q2_K";
    case GGML_TYPE_Q3_K:    return "Q3_K";
    case GGML_TYPE_Q4_K:    return "Q4_K";
    case GGML_TYPE_Q5_K:    return "Q5_K";
    case GGML_TYPE_Q6_K:    return "Q6_K";
    case GGML_TYPE_Q8_K:    return "Q8_K";
    case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
    case GGML_TYPE_IQ2_XS:  return "IQ2_XS";
    case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
    case GGML_TYPE_IQ1_S:   return "IQ1_S";
    case GGML_TYPE_IQ4_NL:  return "IQ4_NL";
    case GGML_TYPE_IQ3_S:   return "IQ3_S";
    case GGML_TYPE_IQ2_S:   return "IQ2_S";
    case GGML_TYPE_IQ4_XS:  return "IQ4_XS";
    case GGML_TYPE_I8:      return "I8";
    case GGML_TYPE_I16:     return "I16";
    case GGML_TYPE_I32:     return "I32";
    case GGML_TYPE_I64:     return "I64";
    case GGML_TYPE_F64:     return "F64";
    case GGML_TYPE_IQ1_M:   return "IQ1_M";
    case GGML_TYPE_BF16:    return "BF16";
    case GGML_TYPE_TQ1_0:   return "TQ1_0";
    case GGML_TYPE_TQ2_0:   return "TQ2_0";
    case GGML_TYPE_MXFP4:   return "MXFP4";
    case GGML_TYPE_NVFP4:   return "NVFP4";
    case GGML_TYPE_Q1_0:    return "Q1_0";
    case GGML_TYPE_Q2_0:    return "Q2_0";
    default:                return "UNKNOWN";
    }
}