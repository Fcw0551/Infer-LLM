#pragma once 

//model name……
struct HParams {
    std::string arch;              // "llama", "internlm2", "qwen2"
    std::string name;            
    uint32_t n_vocab = 0;          // 词表大小（决定 embedding 行数）
    uint32_t n_embd = 0;           // 隐藏层维度
    uint32_t n_layer = 0;          // Transformer 层数
    uint32_t n_head = 0;           // 注意力头数
    uint32_t n_head_kv = 0;        // KV 头数（GQA/MQA，等于 n_head 就是 MHA）
    uint32_t n_ff = 0;             // FFN 中间维度
    //位置编码 
    uint32_t n_ctx_train = 0;      // 训练时最大上下文长度
    float    rope_theta = 10000.0f;
    //归一化
    float    rms_norm_eps = 1e-5f;
    // 可选/扩展字段 
    uint32_t n_embd_head = 0;      // 每个头的维度，通常 = n_embd / n_head
};

struct LayerWeights {
    //----------注意力---------
    Tensor* attn_norm = nullptr;    // "blk.N.attn_norm.weight"
    Tensor* wq = nullptr;           // "blk.N.attn_q.weight"
    Tensor* wk = nullptr;           // "blk.N.attn_k.weight"
    Tensor* wv = nullptr;           // "blk.N.attn_v.weight"
    Tensor* wo = nullptr;           // "blk.N.attn_output.weight"

    // -------------可选：Q/K Norm-----------（Qwen2、InternLM2 有）
    Tensor* attn_q_norm = nullptr;  // "blk.N.attn_q_norm.weight"
    Tensor* attn_k_norm = nullptr;  // "blk.N.attn_k_norm.weight"

    // ------------FFN ----------
    Tensor* ffn_norm = nullptr;     // "blk.N.ffn_norm.weight"
    Tensor* ffn_gate = nullptr;     // "blk.N.ffn_gate.weight"（LLaMA 系）
    Tensor* ffn_up   = nullptr;     // "blk.N.ffn_up.weight"
    Tensor* ffn_down = nullptr;     // "blk.N.ffn_down.weight"
};

struct Weights {
    //------------ 全局权重 ----------
    Tensor* token_embd  = nullptr;  // "token_embd.weight"
    Tensor* output_norm = nullptr;  // "output_norm.weight"
    Tensor* output      = nullptr;  // "output.weight"（有时和 token_embd 共享）

    // ---------- 每层权重 ----------
    std::vector<LayerWeights> layers;   // 大小 = hparams.n_layer
    std::unordered_map<std::string, Tensor*> map;

//---------------工具函数---------------
    Tensor* get(std::string_view name) const {
        auto it = map.find(std::string(name));
        return it == map.end() ? nullptr : it->second;
    }
};

class ModelBase{
private:
    GGUFContext _gguf;                                 //GGUF文件
    HParams _hparams;                                  //超参数  ：模型是什么
    Weights _weights;                                  //权重视图：模型有什么   
    std::vector<std::unique_ptr<Tensor>> _tensor_pool; //Tensor 所有权
public:
    //加载超参数->整体架构
    std::expected<void,std::error_code> load_hparams(const GGUFContext info);
    //加载权重
    std::expected<void,std::error_code> load_weight(const GGUFContext info);
    //不同的模型构建计算图的方式不同
    virtual std::expected<Graph, std::error_code> build_graph(GraphContext& ctx) = 0;
};
//model的工厂方法
class ModelFactory{

};
//具体的模型去继承ModelBase
class Qwen3Model: public ModelBase{

};