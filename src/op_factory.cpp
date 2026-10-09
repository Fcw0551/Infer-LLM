#include "../include/op_factory.hpp"

// OpFactory :: function

// 不需要想batch，batch由上层去构建即可

// 唯一创建中间张量的地方,生命周期交给graph，graph析构时释放
Tensor *OpFactory::create_tensor(Graph &g, std::string_view name, DataType type, std::initializer_list<int64_t> shape){
    auto t = make_unique<Tensor>(name, type, shape);
    return g.add_node(std::move(t));
}

// 基础算子
Tensor *OpFactory::add(Graph &g, Tensor *a, Tensor *b, std::string_view name){
    // 维度肯定要ab一致
    assert(a->dims[0] == b->dims[0]);
    assert(a->dims[1] == b->dims[1]);
    assert(a->dims[2] == b->dims[2]);
    assert(a->dims[3] == b->dims[3]);
    assert(a->dataType == b->dataType);

    Tensor *result = create_tensor(g, name, a->dataType, {a->dims[0], a->dims[1], a->dims[2], a->dims[3]});
    result->op = OperationType::GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}
Tensor *OpFactory::sub(Graph &g, Tensor *a, Tensor *b, std::string_view name){
    // 维度肯定要ab一致
    assert(a->dims[0] == b->dims[0]);
    assert(a->dims[1] == b->dims[1]);
    assert(a->dims[2] == b->dims[2]);
    assert(a->dims[3] == b->dims[3]);
    assert(a->dataType == b->dataType);

    Tensor *result = create_tensor(g, name, a->dataType, {a->dims[0], a->dims[1], a->dims[2], a->dims[3]});
    result->op = OperationType::GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}
Tensor *OpFactory::mul(Graph &g, Tensor *a, Tensor *b, std::string_view name){
    // 维度肯定要ab一致
    assert(a->dims[0] == b->dims[0]);
    assert(a->dims[1] == b->dims[1]);
    assert(a->dims[2] == b->dims[2]);
    assert(a->dims[3] == b->dims[3]);
    assert(a->dataType == b->dataType);

    Tensor *result = create_tensor(g, name, a->dataType, {a->dims[0], a->dims[1], a->dims[2], a->dims[3]});
    result->op = OperationType::GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}
Tensor *OpFactory::div(Graph &g, Tensor *a, Tensor *b, std::string_view name){
    // 维度肯定要ab一致
    assert(a->dims[0] == b->dims[0]);
    assert(a->dims[1] == b->dims[1]);
    assert(a->dims[2] == b->dims[2]);
    assert(a->dims[3] == b->dims[3]);
    assert(a->dataType == b->dataType);

    Tensor *result = create_tensor(g, name, a->dataType, {a->dims[0], a->dims[1], a->dims[2], a->dims[3]});
    result->op = OperationType::GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// embedding查询
Tensor *OpFactory::embedding(Graph &g, Tensor *input_token, Tensor *token_embd){

    // input_token : [n_tokens]
    // token_embd  : [n_embd , n_vocab]
    // embedding   : [n_embd , n_tokens]
    uint64_t n_embd = token_embd->dims[0];
    uint64_t n_tokens = input_token->dims[0];

    Tensor *result = create_tensor(g, "embedding", token_embd->dataType, {n_embd, n_tokens});
    result->op = OperationType::GGML_OP_GET_ROWS;
    result->src[0] = input_token;
    result->src[1] = token_embd;
    return result;
}

// 矩阵乘
// w: [n_embd, n_out], x: [n_embd, n_tokens] → out: [n_out, n_tokens]
Tensor *OpFactory::mul_mat(Graph &g, Tensor *w, Tensor *x, std::string_view name){
    uint64_t n_out = w->dims[1];
    uint64_t n_tokens = x->dims[1];
    Tensor *result = create_tensor(g, name, x->dataType, {n_out, n_tokens});
    result->op = OperationType::GGML_OP_MUL_MAT;
    result->src[0] = w;
    result->src[1] = x;
    return result;
}

// 归一化 : 求平方和 开根号 乘以gamma
Tensor *OpFactory::rms_norm(Graph &g, Tensor *x, Tensor *weight, float eps, std::string_view name){
    // x     :[n_embd , n_tokens]
    // weight:[n_embd,1,1,1]
    // rms_norm_i : [n_embd , n_tokens]
    Tensor *result = create_tensor(g, name, x->dataType, {x->dims[0], x->dims[1], x->dims[2], x->dims[3]});
    result->op = OperationType::GGML_OP_RMS_NORM;
    result->src[0] = x;
    result->src[1] = weight;
    std::memcpy(&result->op_params[0], &eps, sizeof(eps));
    return result;
}

//Tensor *OpFactory::layer_norm(Graph &g, Tensor *x, Tensor *weight, Tensor *bias, float eps);

// 激活
Tensor *OpFactory::silu(Graph &g, Tensor *x, std::string_view name){
    Tensor *result = create_tensor(g, name, x->dataType, {x->dims[0], x->dims[1], x->dims[2], x->dims[3]});
    result->op = OperationType::GGML_OP_UNARY;
    result->src[0] = x;
    uint32_t sub = static_cast<uint32_t>(OperationType_unary_op::GGML_UNARY_OP_SILU);
    std::memcpy(&result->op_params[0], &sub, 4);
    return result;
}
//Tensor *OpFactory::gelu(Graph &g, Tensor *x);

// 位置编码
Tensor *OpFactory::rope(Graph &g, Tensor *x, Tensor *inp_pos, Tensor *rope_theta_table, std::string_view name){
    // 输入和输出形状一致
    Tensor *result = create_tensor(g, name, x->dataType, {x->dims[0], x->dims[1], x->dims[2], x->dims[3]});
    result->op = OperationType::GGML_OP_ROPE;
    result->src[0] = x;
    result->src[1] = inp_pos;
    result->src[2] = rope_theta_table;
    return result;
}

// 注意力
//Tensor *OpFactory::soft_max(Graph &g, Tensor *x, Tensor *mask, float scale);
Tensor *OpFactory::flash_attn(Graph &g, Tensor *q, Tensor *k, Tensor *v, const AttnParams &p, std::string_view name){
    // q[n_embd,n_tokens]
    // k[n_embd,n_tokens]
    // v[n_embd,n_tokens]
    Tensor *result = create_tensor(g, name, q->dataType, {q->dims[0], q->dims[1], 1, 1});
    result->op = OperationType::GGML_OP_FLASH_ATTN_EXT;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    return result;
}

// 索引
// table: [n_embd, n_vocab], indices: [n_tokens] → out: [n_embd, n_tokens]
//Tensor *OpFactory::get_rows(Graph &g, Tensor *table, Tensor *indices);

// 形状
//Tensor *OpFactory::reshape(Graph &g, Tensor *x, std::initializer_list<int64_t> shape);
//Tensor *OpFactory::permute(Graph &g, Tensor *x, std::initializer_list<int> axes);
//Tensor *OpFactory::cont(Graph &g, Tensor *x); // 强制连续
//Tensor *OpFactory::view(Graph &g, Tensor *x, std::initializer_list<int64_t> shape);
