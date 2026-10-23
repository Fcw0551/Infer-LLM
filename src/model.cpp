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
        (*embd_exp)->layer=HParams.n_layer;
        (*embd_exp)->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
        _weights.token_embd = *embd_exp;
        auto out_norm_exp = create_tensor("output_norm.weight");
        if (!out_norm_exp) return std::unexpected(out_norm_exp.error());
        (*out_norm_exp)->layer=HParams.n_layer;
        (*out_norm_exp)->tensorRole=TensorRole::TENSOR_ROLE_WEIGHT;
        _weights.output_norm = *out_norm_exp;

        auto out_exp = create_tensor("output.weight");
        if (!out_exp) {
            _weights.output = _weights.token_embd;  // Qwen3 等 weight tying 模型：output.weight == token_embd.weight
        }
        else{
            (*out_exp)->layer = HParams.n_layer;
            (*out_exp)->tensorRole = TensorRole::TENSOR_ROLE_WEIGHT;
            _weights.output = *out_exp;
        }
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


// load_weight之后执行，把每层的weight安排的device放好，后续scheduler能够进行划分图and调度 
// tips：这里会再确定好权重的backend之后，开辟权重内存，之前mmap只是映射未占用实际的物理内存
std::expected<void, std::error_code> ModelBase::plan_weights_placement(const std::vecotr<Backend*>& devices,const struct PlacementConfig&cfg){
    //  // 推理配置 ： 方便后续计算图的安排device
    // struct PlacementConfig {
    //     uint32_t n_ctx = 0;                     // 本次推理上下文长度
    //     size_t kv_elem_size = 4;                // v1 按 f32 估
    //     size_t margin_bytes  = 1ull << 30;      // 每设备预留
    //     size_t workspace_bytes = 256ull << 20;  //256MB
    // };
    // //device的配置
    // struct Cap{
    //     Backend *dev; // 设备
    //     size_t left;  // 余量
    // };

    // 检查
    if (devices.empty()){
        return std::expected(std::make_error_code(std::errc::invalid_argument));
    }
    if (_weights.layers.empty()){
        return std::expected(std::make_error_code(std::errc::invalid_argument));
    }
    if (HParams.n_layer == 0 || HParams.n_head_kv == 0){
        return std::expected(std::make_error_code(std::errc::invalid_argument));
    }
    if (cfg.n_ctx == 0){
        return std::expected(std::make_error_code(std::errc::invalid_argument));
    }

    //devices 的顺序就是优先级：{cuda:0, cuda:1, cpu}。多 GPU 时层会按这个顺序 first-fit，后放的层落在靠前的 GPU 上。

    // 从后往前放 一层一层放 最后必须要有cpu兜底



    //-------------- 初始化设备余量 ------------
    bool has_host=false;
    Backend* cpu_backend = nullptr;
    std::vector<Cap> caps;

    for(auto &device:devices){
        // 找有没有cpu
        if(device->is_host()){
            has_host=true;
            cpu_backend = device;
        }
        // 计算每台设备能不能运行 ： 扣除 margin_bytes + workspace_bytes (运行时开销)
        size_t total = device->get_info().memory_bytes;
        if (total < cfg.margin_bytes + cfg.workspace_bytes) // 有隐患，可能用户传一张很小的卡+cpu，cpu能跑，但是传了gpu导致后续都跑不了
            return std::expected(std::make_error_code(std::errc::not_enough_memory));
        caps.push_back({ device, total - cfg.margin_bytes - cfg.workspace_bytes });

    }
    if(!has_host){
        return std::expected(std::make_error_code(std::errc::not_supported));
    }

    // intput固定放在cpu了，因为本质就是索引

    // --------------估算每层放置的开销-----------
    std::vector<size_t> cost_layer(HParams.n_layer);
    // kv的隐藏层：HParams.n_head_kv * HParams.n_embd_head 
    size_t kv_size_per_layer = 2ull * cfg.n_ctx * HParams.n_head_kv * HParams.n_embd_head * cfg.kv_elem_size;
    // 权重大小 lambda表达式辅助查询
    auto accumulate_layer=[&](const LayerWeights& lw)->size_t {
        size_t sum=lw.attn_norm->size_bytes()+lw.wq->size_bytes()+lw.wk->size_bytes()+lw.wv->size_bytes()+
        lw.wo->size_bytes()+lw.ffn_norm->size_bytes()+lw.ffn_gate->size_bytes()+lw.ffn_up->size_bytes()+lw.ffn_down->size_bytes();
        if(lw.attn_q_norm){
            sum+=lw.attn_q_norm->size();
        }
        if(lw.attn_k_norm){
            sum+=lw.attn_k_norm->size();
        }
        return sum;
    };
    for(size_t i=0;i<cost_layer.size();++i){
        cost_layer[i]=kv_size_per_layer+accumulate_layer[i];
    }

    // 输出层的开销 放在最后面了，可以单独的去放在cpu或者gpu（推荐优先gpu和最后一层输出一起）
    const bool tied = (_weights.output == _weights.token_embd);
    const size_t norm_bytes = _weights.output_norm->byte_size();
    const size_t out_w_bytes = tied ? _weights.token_embd->byte_size() : _weights.output->byte_size();

    // ----------- 贪心放置 -----------
    auto try_place_gpu = [&](size_t cost) -> Backend *{
        for (auto &c : caps)
        {
            if (c.dev->is_host())
                continue; // CPU 不参与主动放置
            if (c.left >= cost){
                c.left -= cost;
                return c.dev;
            }
        }
        return nullptr;
    };
    // 默认全部都是cpu
    std::vector<Backend*> dev_layer(HParams.n_layer,cpu_backend);
    
    // output 单元 这里如果output放在cpu，可能效果很差
    Backend *dev_output try_place_gpu(out_w_bytes); 
    if (dev_output && tied){
        // 真正决定上 GPU 了才补副本（元信息，同一 GGUF offset）
        auto dup = std::make_unique<Tensor>(_gguf.find_tensor("token_embd.weight"));    // 根据tensorinfo去构建一个tensor对象
        dup->name = "output.weight";
        dup->layer = _hparams.n_layer;
        dup->tensorRole = TensorRole::TENSOR_ROLE_WEIGHT;
        _weights.output = dup.get();
        _weights.map["output.weight"] = dup.get();
        _tensor_pool.push_back(std::move(dup));
    }
   
    // transformer 层，从后往前
    bool gpu_run = true;
    for (int il = HParams.n_layer - 1; il >= 0; --il){
        if (!gpu_run)
            break; // 关键：一旦失败就停：不要继续往前搜索可行解了，否则会导致断层，gpu cpu gpu ，目的是cpu cpu cpu …… gpu gpu gpu
        if (Backend *d = try_place_gpu(layer_cost[il]))
            dev_layer[il] = d;
        else
            gpu_run = false;
    }
   
    // cpu兜底 现在计算一下cpu够不够 
    size_t cpu_need=0;
    if(dev_output==nullptr){
        dev_output=cpu_backend;
        // 落 CPU 时不复制：tied 的话权重已随 token_embd 计过，只加 norm
        cpu_need += norm_bytes + (tied ? 0 : out_w_bytes);
    }
    for (int il = 0; il < n_layer; ++il)
        if (dev_layer[il] == cpu)
            cpu_need += layer_cost[il];
    // 还要加上输入的
    cpu_need += _weights.token_embd->byte_size();
    size_t cpu_cap=cpu_backend->get_info().memory_bytes - cfg.margin_bytes - cfg.workspace_bytes;
    if(cpu_cap<cpu_need){
        return std::expected(std::make_error_code(std::errc::not_enough_memory));
    }

    // --------- 放置完成 -----------

    _dev_layer = std::move(dev_layer);
    _dev_output = dev_output;
    // 辅助每个tensor标记backend
    auto bind = [](Tensor *t, Backend *b)
    {
        if (!t)
            return;
        t->backend = b;
        t->device = b->get_info().type;
    };
   
    // 辅助遍历每层的weight
    for (int il = 0; il < n_layer; ++il)
        for (Tensor *t : get_layer_tensors(_weights.layers[il]))
            bind(t, _dev_layer[il]);

    bind(_weights.token_embd, cpu); // input 永远 CPU
    bind(_weights.output_norm, _dev_output);
    bind(_weights.output, _dev_output);
    
    // 开辟实际占用的内存空间，加载weight
    auto result=alloc_weight();
    if(!result){
        return std::unexpected(result.error());
    }
    
    return {};

}


// 私有：放置确定后调用，把权重绑定到 _dev_layer / _dev_output / _dev_input
std::expected<void, std::error_code> ModelBase::alloc_weight(){
    // 按设备分组
    std::unordered_map<Backend *, std::vector<Tensor *>> groups;
    auto add = [&](Tensor *t, Backend *dev){
        if (!t || !dev) return;
        groups[dev].push_back(t);
    };

    add(_weights.token_embd, _dev_input);
    add(_weights.output_norm, _dev_output);
    add(_weights.output,     _dev_output);
    for (int il = 0; il < (int)_dev_layer.size(); ++il)
        for (Tensor *t : get_layer_tensors(_weights.layers[il]))
            add(t, _dev_layer[il]);

    // 每个 tensor 在文件里的地址（GGUF 的 offset 相对数据区）
    auto src_of = [&](const Tensor *t){
        // mmap文件的起始位置 + 数据区偏移 + tensor的偏移 = tensor的起始
        return _gguf.data_base() + _gguf.data_region_offset() + t->offset;
    };

    constexpr size_t ALIGN = 64;                       // 向上对齐数 TODO：应该由backend自动探测，往上报
    auto align_up = [](size_t x, size_t a){ return (x + a - 1) / a * a; };

    // host：直接绑定，不分配
    // 非 host：算总大小，开一块
    struct Arena { Backend *backend; size_t bytes; void *base; };
    std::vector<Arena> arenas;

    // 失败时统一回滚：只回收本次已经开出来的 arena
    // host没有加入arena，释放非host即可
    auto rollback = [&]() noexcept {
        for (const auto &a : arenas)
            for (Tensor *t : groups.at(a.backend))
                t->data = nullptr;                          // 尚未切分时本来就是 nullptr，重复置空无害
        for (const auto &a : arenas)
            a.backend->free(BufferKind::WEIGHTS, a.base);
        arenas.clear();
    };

    // 开辟逻辑
    for (auto &[dev, tv] : groups) {
        // 如果是host
        if (dev->is_host()) {
            // 遍历vector里面的每个tensor，把data指向即可，因为是host之前映射过 
            for (Tensor *t : tv)
                t->data = const_cast<std::byte *>(src_of(t));
            continue;
        }
        // 如果是非host
        size_t total = 0;
        // 统计一共多少
        for (Tensor *t : tv)
            total = align_up(total, ALIGN) + t->byte_size();
        // 统一开辟
        auto p = dev->alloc(BufferKind::WEIGHTS, total);
        if (!p) { rollback(); return std::unexpected(p.error()); }
        arenas.push_back({dev, total, *p});                         // 这里是记录一个backend开辟了多少，初始的指针是在哪，释放的时候要传p
    }

    // 切分
    for (auto &a : arenas) {
        size_t off = 0;
        // arenas里面的backend，去groups里面对应找到tensor
        for (Tensor *t : groups[a.dev]) {
            off = align_up(off, ALIGN);
            t->data = static_cast<char *>(a.base) + off;
            off += t->byte_size();
        }
    }

    // 上传（只对非 host）
    for (auto &a : arenas)
        for (Tensor *t : groups[a.dev]) {
            // 直接调用host-> 非host拷贝函数即可
            // t->data是之前开辟好的非host空间
            auto r = a.dev->copy_tenosr_set(t, src_of(t), 0, t->byte_size());
            if (!r){
                // 回滚
                rollback();
                return r;
            }
        }

    // 记下来，供析构/重载时整块释放
    for (auto &a : arenas) _weight_arenas.push_back({a.dev, a.base});
    return {};
}

std::vector<Tensor *> ModelBase::get_layer_tensors(const LayerWeights &lw){
    std::vector<Tensor *> ts;
    ts.reserve(11); // 预分配，避免扩容

    // 把指针装进去，后面在 bind 的时候再统一判空
    ts.push_back(lw.attn_norm);
    ts.push_back(lw.wq);
    ts.push_back(lw.wk);
    ts.push_back(lw.wv);
    ts.push_back(lw.wo);
    ts.push_back(lw.ffn_norm);
    ts.push_back(lw.ffn_gate);
    ts.push_back(lw.ffn_up);
    ts.push_back(lw.ffn_down);

    if (lw.attn_q_norm)
        ts.push_back(lw.attn_q_norm);
    if (lw.attn_k_norm)
        ts.push_back(lw.attn_k_norm);

    return ts;
};

//----------------------注册表------------------------
void register_Qwen3(){
    ModelRegistry::instance().register_arch("qwen3",
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
