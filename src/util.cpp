#include "../include/util.hpp"
// ============ 枚举转字符串 ============
const char *gguf_value_type_name(GGUFType t){
    switch (t){
    case GGUFType::GGUF_TYPE_UINT8:
        return "UINT8";
    case GGUFType::GGUF_TYPE_INT8:
        return "INT8";
    case GGUFType::GGUF_TYPE_UINT16:
        return "UINT16";
    case GGUFType::GGUF_TYPE_INT16:
        return "INT16";
    case GGUFType::GGUF_TYPE_UINT32:
        return "UINT32";
    case GGUFType::GGUF_TYPE_INT32:
        return "INT32";
    case GGUFType::GGUF_TYPE_UINT64:
        return "UINT64";
    case GGUFType::GGUF_TYPE_INT64:
        return "INT64";
    case GGUFType::GGUF_TYPE_FLOAT32:
        return "FLOAT32";
    case GGUFType::GGUF_TYPE_FLOAT64:
        return "FLOAT64";
    case GGUFType::GGUF_TYPE_BOOL:
        return "BOOL";
    case GGUFType::GGUF_TYPE_STRING:
        return "STRING";
    case GGUFType::GGUF_TYPE_ARRAY:
        return "ARRAY";
    default:
        return "UNKNOWN";
    }
}

const char* ggml_type_name(DataType t) {
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


size_t dataType_size(DataType dataType){
    switch (dataType)
    {
        case GGML_TYPE_F32:     return 4;
        case GGML_TYPE_F16:     return 2;
        case GGML_TYPE_Q4_0:    return 18;
        case GGML_TYPE_Q4_1:    return 20;
        case GGML_TYPE_Q5_0:    return 22;
        case GGML_TYPE_Q5_1:    return 24;
        case GGML_TYPE_Q8_0:    return 34;
        case GGML_TYPE_Q8_1:    return 40;
        case GGML_TYPE_Q2_K:    return 14;
        case GGML_TYPE_Q3_K:    return 18;
        case GGML_TYPE_Q4_K:    return 20;
        case GGML_TYPE_Q5_K:    return 22;
        case GGML_TYPE_Q6_K:    return 24;
        case GGML_TYPE_Q8_K:    return 32;
        case GGML_TYPE_IQ2_XXS: return 10;
        case GGML_TYPE_IQ2_XS:  return 12;
        case GGML_TYPE_IQ3_XXS: return 18;
        case GGML_TYPE_IQ1_S:   return 10;
        case GGML_TYPE_IQ4_NL:  return 18;
        case GGML_TYPE_IQ3_S:   return 18;
        case GGML_TYPE_IQ2_S:   return 14;
        case GGML_TYPE_IQ4_XS:  return 22;
        case GGML_TYPE_I8:      return 1;
        case GGML_TYPE_I16:     return 2;
        case GGML_TYPE_I32:     return 4;
        case GGML_TYPE_I64:     return 8;
        case GGML_TYPE_F64:     return 8;
        case GGML_TYPE_IQ1_M:   return 8;
        case GGML_TYPE_BF16:    return 2;
        case GGML_TYPE_TQ1_0:   return 14;
        case GGML_TYPE_TQ2_0:   return 16;
        case GGML_TYPE_MXFP4:   return 8;
        case GGML_TYPE_NVFP4:   return 32;
        case GGML_TYPE_Q1_0:    return 8;
        case GGML_TYPE_Q2_0:    return 10;

        case GGML_TYPE_COUNT: // COUNT只是枚举边界，非法
        default:
            return 0;
    }
}