#include "../include/model.hpp"
#include "../include/models/internLm2.hpp"
#include "../include/models/Qwen3.hpp"

//创建全局单例
ModelRegistry& ModelRegistry::instance(){
    static ModelRegistry reg;   
    return reg;
   
}
//注册：架构名 - 创建函数
void ModelRegistry::register_arch(std::string arch, Creator creator){
    _registry[std::move(arch)]=std::move(creator);
}

// 根据架构名找到模型
std::expected<ModelRegistry::Creator, std::error_code> ModelRegistry::create(const std::string& arch) const{
    auto it=_registry.find(arch);
    if(it==_registry.end()){
        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }
    return it->second;
}

//列出所有已注册的架构
std::vector<std::string> ModelRegistry::supported_arches() const{
    std::vector<std::string> result;
    result.reserve(_registry.size());
    for (const auto& [arch,creator] : _registry) result.push_back(arch);
    return std::move(result);
}



//------------------------ModelBase--------------------
std::expected<std::unique_ptr<ModelBase>, std::error_code> ModelBase::load_model(GGUFContext&& gguf){
    //先查模型的架构
    auto arch=gguf.metadata().get_meta<std::string>("general.architecture");
    if(!arch){
        return std::unexpected(arch.error());
    }
    auto Creator=ModelRegistry::instance().create(*arch);
    if(!Creator){
        return std::unexpected(Creator.error());
    }
    return (*Creator)(std::move(gguf));
}



std::expected<void,std::error_code> ModelBase::load_hparams(){
    // --- 读取基础全局KV ---
    auto arch_exp = _gguf.metadata().get_meta<std::string>("general.architecture");
    if (!arch_exp) {
        return std::unexpected(arch_exp.error());
    }

    auto name_exp = _gguf.metadata().get_meta<std::string>("general.name");
    if (!name_exp) {
        return std::unexpected(name_exp.error());
    }

    // 词表大小：tokenizer.ggml.tokens 数组长度
    auto n_vocab_exp = _gguf.metadata().get_meta_array_len("tokenizer.ggml.tokens");
    if (!n_vocab_exp) {
        return std::unexpected(n_vocab_exp.error());
    }

    _hparams.arch = *arch_exp;
    _hparams.name = *name_exp;
    _hparams.n_vocab = static_cast<uint32_t>(*n_vocab_exp);


    // 根据架构拼接前缀，读取模型参数 ---
    const std::string& prefix = _hparams.arch;

    //embedding_length 隐藏层维度 n_embd
    auto n_embd_exp = _gguf.metadata().get_meta<uint32_t>(prefix + ".embedding_length");
    if (!n_embd_exp) return std::unexpected(n_embd_exp.error());

    //block_count 层数
    auto n_layer_exp = _gguf.metadata().get_meta<uint32_t>(prefix + ".block_count");
    if (!n_layer_exp) return std::unexpected(n_layer_exp.error());
  
    //attention.head_count总注意力头 n_head
    auto n_head_exp = _gguf.metadata().get_meta<uint32_t>(prefix + ".attention.head_count");
    if (!n_head_exp) return std::unexpected(n_head_exp.error());

    //attention.head_count_kv KV头 n_head_kv
    auto n_head_kv_exp = _gguf.metadata().get_meta<uint32_t>(prefix + ".attention.head_count_kv");
    if (!n_head_kv_exp) return std::unexpected(n_head_kv_exp.error());

    //feed_forward_length FFN维度 n_ff
    auto n_ff_exp = _gguf.metadata().get_meta<uint32_t>(prefix + ".feed_forward_length");
    if (!n_ff_exp) return std::unexpected(n_ff_exp.error());

    // context_length 训练上下文窗口 n_ctx_train
    auto n_ctx_exp = _gguf.metadata().get_meta<uint32_t>(prefix + ".context_length");
    if (!n_ctx_exp) return std::unexpected(n_ctx_exp.error());

    // rope.freq_base → rope_theta
    auto rope_theta_exp = _gguf.metadata().get_meta<float>(prefix + ".rope.freq_base");
    if (rope_theta_exp) {
        _hparams.rope_theta = *rope_theta_exp;
    } else {
        // 读取失败就用默认值10000.0
        _hparams.rope_theta = 10000.0f;
    }

    // attention.layer_norm_rms_epsilon
    auto rms_eps_exp = _gguf.metadata().get_meta<float>(prefix + ".attention.layer_norm_rms_epsilon");
    if (rms_eps_exp) {
        _hparams.rms_norm_eps = *rms_eps_exp;
    } else {
        _hparams.rms_norm_eps = 1e-5f;
    }

    _hparams.n_embd = *n_embd_exp;
    _hparams.n_layer = *n_layer_exp;
    _hparams.n_head = *n_head_exp;
    _hparams.n_head_kv = *n_head_kv_exp;
    _hparams.n_ctx_train = *n_ctx_exp;
    _hparams.n_ff = *n_ff_exp;

    //---自动计算每个头维度 n_embd_head ---
    if (_hparams.n_head == 0) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    _hparams.n_embd_head = _hparams.n_embd / _hparams.n_head;

    // 参数合法性校验---
    if (_hparams.n_embd == 0 || _hparams.n_layer ==0 || _hparams.n_vocab ==0) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (_hparams.n_head_kv > _hparams.n_head) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    return {}; //成功 std::expected<void> 返回空
  
}
std::expected<void,std::error_code> ModelBase::load_weight(){

    //初始化Weights的layers数组，层数等于_hparams.n_layer
    _weights.layers.resize(_hparams.n_layer);

    //辅助lambda：根据名字从GGUF找TensorInfo，创建Tensor，存入pool+weights.map
    auto create_tensor = [this](std::string_view tensor_name) -> std::expected<Tensor*, std::error_code>
    {
        //从GGUF的索引查找张量元信息
        TensorInfo* info = _gguf.find_tensor(std::string(tensor_name));
        if (!info){
            return std::unexpected(std::make_error_code(std::errc::no_such_file_or_directory));
        }

        //根据TensorInfo构造Tensor对象，加入tensor_pool
        auto tensor_ptr = std::make_unique<Tensor>(info);
        Tensor* raw_ptr = tensor_ptr.get();

        //转移所有权进pool
        _tensor_pool.push_back(std::move(tensor_ptr));
        // 存入Weights全局名字map
        _weights.map[std::string(tensor_name)] = raw_ptr;

        return raw_ptr;
    };

    //全局权重
    {
        auto embd_exp = create_tensor("token_embd.weight");
        if (!embd_exp) return std::unexpected(embd_exp.error());
        _weights.token_embd = *embd_exp;

        auto out_norm_exp = create_tensor("output_norm.weight");
        if (!out_norm_exp) return std::unexpected(out_norm_exp.error());
        _weights.output_norm = *out_norm_exp;

        auto out_exp = create_tensor("output.weight");
        if (!out_exp) return std::unexpected(out_exp.error());
        _weights.output = *out_exp;
    }

    //逐层加载每一层LayerWeights
    for (size_t layer_idx = 0; layer_idx < _hparams.n_layer; ++layer_idx)
    {
        auto& lw = _weights.layers[layer_idx];
        std::string prefix = "blk." + std::to_string(layer_idx) + ".";

        // Attention
        {
            auto q_exp = create_tensor(prefix + "attn_q.weight");
            if (!q_exp) return std::unexpected(q_exp.error());
            (*q_exp)->layer=layer_idx;
            (*q_exp)->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
            lw.wq = *q_exp;
            

            auto k_exp = create_tensor(prefix + "attn_k.weight");
            if (!k_exp) return std::unexpected(k_exp.error());
            (*k_exp)->layer=layer_idx;
            (*k_exp)->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
            lw.wk = *k_exp;

            auto v_exp = create_tensor(prefix + "attn_v.weight");
            if (!v_exp) return std::unexpected(v_exp.error());
            (*v_exp)->layer=layer_idx;
            (*v_exp)->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
            lw.wv = *v_exp;

            auto wo_exp = create_tensor(prefix + "attn_output.weight");
            if (!wo_exp) return std::unexpected(wo_exp.error());
            (*wo_exp)->layer=layer_idx;
            (*wo_exp)->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
            lw.wo = *wo_exp;

            auto attn_norm_exp = create_tensor(prefix + "attn_norm.weight");
            if (!attn_norm_exp) return std::unexpected(attn_norm_exp.error());
            (*attn_norm_exp)->layer=layer_idx;
            (*attn_norm_exp)->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
            lw.attn_norm = *attn_norm_exp;
        }

        // FFN
        {
            auto ffn_norm_exp = create_tensor(prefix + "ffn_norm.weight");
            if (!ffn_norm_exp) return std::unexpected(ffn_norm_exp.error());
            (*ffn_norm_exp)->layer=layer_idx;
            (*ffn_norm_exp)->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
            lw.ffn_norm = *ffn_norm_exp;

            auto gate_exp = create_tensor(prefix + "ffn_gate.weight");
            if (!gate_exp) return std::unexpected(gate_exp.error());
            (*gate_exp)->layer=layer_idx;
            (*gate_exp)->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
            lw.ffn_gate = *gate_exp;

            auto up_exp = create_tensor(prefix + "ffn_up.weight");
            if (!up_exp) return std::unexpected(up_exp.error());
            (*up_exp)->layer=layer_idx;
            (*up_exp)->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
            lw.ffn_up = *up_exp;

            auto down_exp = create_tensor(prefix + "ffn_down.weight");
            if (!down_exp) return std::unexpected(down_exp.error());
            (*down_exp)->layer=layer_idx;
            (*down_exp)->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
            lw.ffn_down = *down_exp;
        }

        //特殊张量
        Tensor *qn = nullptr;
        TensorInfo *qn_info = _gguf.find_tensor(prefix + "attn_q_norm.weight");
        if (qn_info)
        {
            auto exp = create_tensor(prefix + "attn_q_norm.weight");
            if (!exp)
                return std::unexpected(exp.error());
            qn = *exp;
            qn->layer=layer_idx;
            qn->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
        }
        lw.attn_q_norm = qn;

        Tensor *kn = nullptr;
        TensorInfo *kn_info = _gguf.find_tensor(prefix + "attn_k_norm.weight");
        if (kn_info)
        {
            auto exp = create_tensor(prefix + "attn_k_norm.weight");
            if (!exp)
                return std::unexpected(exp.error());
            kn = *exp;
            kn->layer=layer_idx;
            kn->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
        }
        lw.attn_k_norm = kn;

    }

    return {};
}





//----------------------注册表------------------------
void register_Qwen3(){
    ModelRegistry::instance().register_arch("Qwen3",
        [](GGUFContext&& ctx) -> std::unique_ptr<ModelBase> {
            return std::make_unique<Qwen3Model>(std::move(ctx));
        });
}
void register_internlm2(){
    ModelRegistry::instance().register_arch("internlm2",
        [](GGUFContext&& ctx) -> std::unique_ptr<ModelBase> {
            return std::make_unique<internlm2Model>(std::move(ctx));
        });
}

void register_all_models() {

    register_Qwen3();
    register_internlm2();
    //register_llama();
}
