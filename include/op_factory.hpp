#pragma once
#include "tensor.hpp"
#include "op_params.hpp"
#include "graph.hpp"
#include <initializer_list>
#include <string>
#include <memory>
#include "util.hpp"
#include <iostream>

class OpFactory {
public:
    // 基础算子   
    static Tensor* add(Graph& g, Tensor* a, Tensor* b,std::string_view name="add");
    static Tensor* sub(Graph& g, Tensor* a, Tensor* b,std::string_view name="sub");
    static Tensor* mul(Graph& g, Tensor* a, Tensor* b,std::string_view name="mul");
    static Tensor* div(Graph& g, Tensor* a, Tensor* b,std::string_view name="div");

    //embedding查询
    static Tensor* embedding(Graph&g,Tensor*input_token,Tensor*token_embd);
    // 矩阵
    // w: [n_in, n_out], x: [n_in, n_tokens] → out: [n_out, n_tokens]
    static Tensor* mul_mat(Graph& g, Tensor* w, Tensor* x,std::string_view name="mul_mat");

    // 归一化 
    static Tensor* rms_norm(Graph& g,Tensor* x, Tensor* weight, float eps,std::string_view name="rms_norm");
    static Tensor* layer_norm(Graph& g, Tensor* x, Tensor* weight, Tensor* bias, float eps);

    // 激活
    static Tensor *silu(Graph &g, Tensor *x, std::string_view name = "silu");
    static Tensor *gelu(Graph &g, Tensor *x, std::string_view name = "gelu");

    // 位置编码 
    static Tensor* rope(Graph& g, Tensor* x, Tensor* inp_pos,Tensor* rope_theta_table,std::string_view name="rope");

    // 注意力
    static Tensor* soft_max(Graph& g, Tensor* x, Tensor* mask, float scale);
    // TODO:如果要新增pageattention的话就要新增加接口，或者本身要支持跨步访问
    static Tensor* flash_attn(Graph& g, Tensor* q, Tensor* k, Tensor* v,const AttnParams& p,std::string_view name="flash_attn");

    // 索引
    // table: [n_embd, n_vocab], indices: [n_tokens] → out: [n_embd, n_tokens]
    static Tensor* get_rows(Graph& g, Tensor* table, Tensor* indices);

    // 形状
    static Tensor* reshape(Graph& g, Tensor* x, std::initializer_list<int64_t> shape,std::string_view name="reshape");
    static Tensor* permute(Graph& g, Tensor* x, std::initializer_list<int> axes);
    static Tensor* cont(Graph& g, Tensor* x);    // 强制连续
    static Tensor* view(Graph& g, Tensor* x, std::initializer_list<int64_t> shape);

private:
    // 唯一创建中间张量的地方,生命周期交给graph，graph析构时释放
    static Tensor *create_tensor(Graph &g, std::string_view name, DataType type, std::initializer_list<int64_t> shape);
};

