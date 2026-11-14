// 自己手写，梳理一下架构

#include "../include/backend.hpp"
#include "../include/gguf_parser.hpp"
#include "../include/tokenizer.hpp"
#include "../include/model.hpp"
#include "../include/graph.hpp"
#include "../include/kvcache/kvcacheManager.hpp"
#include "../include/kvcache/KVStrategys/contiguousKVStrategy.hpp"
#include "../include/kvcache/KVStorages/naiveKVStorage.hpp"
#include "../include/scheduler.hpp"
#include "../include/sampler.hpp"
#include <cuda_runtime_api.h>
int main(){
    // ----------------------------- 后端模块 --------------------------------------------
    // 注册后端
    register_all_backends();

    // 加载单例
    BackendRegistry& br = BackendRegistry::instance();

    // 探测后端
    br.probe_all();

    // 探测结果
    br.print();
    // name:         cuda:0
    // DeviceType:   CUDA
    // memory_bytes: 7441743872
    // name:         cpu
    // DeviceType:   CPU
    // memory_bytes: 6060752896

    // 加载后端
    // 内存池配置
    PoolConfig pool_cfg={
        .weights = Strategy::NAIVE,
        .activation = Strategy::NAIVE,
        .kvcache = Strategy::NAIVE
    };
    // 创建cpu后端
    auto result = br.create("cpu",pool_cfg);
    if(!result){
        std::cerr << "创建cpu后端失败：" << result.error().message() << '\n';
        return 1;
    }
    std::unique_ptr<Backend> cpu_backend = std::move(*result);
    // 创建cuda后端
    result = br.create("cuda:0",pool_cfg);
    if(!result){
        std::cerr << "创建cuda后端失败：" << result.error().message() << '\n';
        return 1;
    }
    std::unique_ptr<Backend> gpu_backend = std::move(*result);

    // -------------------------------------- 加载权重 ----------------------------------------
    const std::string gguf_path="/home/wcf/Infer-LLM/models/Qwen3-0.6B-F32.gguf";
    auto result_gguf = GGUFContext::load(gguf_path);
    if (!result_gguf){
        std::cerr << "加载gguf失败：" << result_gguf.error().message() << '\n';
        return 1;
    }
    GGUFContext& gguf_context = *result_gguf;
    gguf_context.print(13);


 
    // -------------------------------------- 初始化tokenizer ----------------------------------
    auto tokenizer = Tokenizer::from_gguf(gguf_context.metadata());
    if (!tokenizer){
        std::cerr << "加载 tokenizer 失败：" << tokenizer.error().message() << '\n';
        return 1;
    }
    // 创建prompt 思考模式
    const std::string_view prompt =
        "<|im_start|>user\n"
        "介绍一下你自己<|im_end|>\n"
        "<|im_start|>assistant\n";
    
    // 非思考模式
    // const std::string_view prompt =
    //     "<|im_start|>user\n"
    //     "介绍一下你自己<|im_end|>\n"
    //     "<|im_start|>assistant\n"
    //     "<think>\n\n</think>\n\n";
    
    auto encoded = tokenizer->encode(prompt, {.parse_special = true});
    if (!encoded) {
        std::cerr << "prompt 编码失败：" << encoded.error().message() << '\n';
        return 1;
    }
    if (encoded->empty()) {
        std::cerr << "prompt 编码结果不能为空\n";
        return 1;
    }
    std::vector<int32_t> token_ids = std::move(*encoded);

    

    // --------------------------------- 加载model ------------------------------------------

    // 以下的参数是后面要用的，避免gguf传给model初始化导致null，所以这里先把后面要用的参数提取出来
    const auto eos_token = gguf_context.metadata().get_meta<uint32_t>("tokenizer.ggml.eos_token_id");
    const auto arch = gguf_context.metadata().get_meta<std::string>("general.architecture");
    const auto n_layers = gguf_context.metadata().get_meta<uint32_t>(*arch + ".block_count");
    const auto n_embd = gguf_context.metadata().get_meta<uint32_t>(*arch + ".embedding_length");
    const auto n_heads = gguf_context.metadata().get_meta<uint32_t>(*arch + ".attention.head_count");
    const auto n_kv_heads = gguf_context.metadata().get_meta<uint32_t>(*arch + ".attention.head_count_kv");
    const auto key_length = gguf_context.metadata().get_meta<uint32_t>(*arch + ".attention.key_length");
    const auto value_length = gguf_context.metadata().get_meta<uint32_t>(*arch + ".attention.value_length");
    const uint32_t k_head_dim = key_length.value_or(*n_embd / *n_heads);
    const uint32_t v_head_dim = value_length.value_or(*n_embd / *n_heads);
    // 注册模型
    register_all_models();
    
    // 加载单例 (这里可以不做，因为所有的已经整合到modelBase当中了)
    ModelRegistry& mr=ModelRegistry::instance();

    //supported_arches
    std::cout<<"支持的架构有"<<std::endl;
    for(auto& arch : mr.supported_arches()){
        std::cout<<arch<<std::endl;
    }
   
    // 加载qwen3
    auto result_qwen3=ModelBase::load_model(std::move(gguf_context));
    if(!result_qwen3){
        std::cerr << "加载 qwen3 失败：" << result_qwen3.error().message() << '\n';
        return 1;
    }
    auto qwen3 = std::move(*result_qwen3);
    // 加载超参数 and 加载权重
    auto result_load_hparams = qwen3->load_hparams();
    auto result_load_weight = qwen3->load_weight();
    if(!result_load_hparams || !result_load_weight){
        std::cerr << "加载 qwen3 hparams || weight 失败：" << result_qwen3.error().message() << '\n';
        return 1;
    }

    // 放置权重
    PlacementConfig cfg_placement={
        .n_ctx = 4096,                      // 支持不了40960这么大，这么大的得一层用8GB左右
        .kv_elem_size = 4,
        .margin_bytes = 1ull << 30,
        .workspace_bytes = 256ull << 20
    };
    //devices 的顺序就是优先级：{cuda:0, cuda:1, cpu}。多 GPU 时层会按这个顺序 first-fit，后放的层落在靠前的 GPU 上
    std::vector<Backend*> devices = {gpu_backend.get(),cpu_backend.get()};
    auto result_plan_weight = qwen3->plan_weights_placement(devices,cfg_placement);
    if(!result_plan_weight){
        std::cerr<<"放置权重失败"<<result_plan_weight.error().message()<<"\n";
        return 1;
    }

    // ------------------------------- kvcache manager --------------------------------
    
    const size_t context_limit = std::min<size_t>(4096, cfg_placement.n_ctx);   // n_ctx是之前配置的缓存空间
    if (token_ids.size() > context_limit){
        std::cerr << "prompt 超过上下文长度限制\n";
        return 1;
    }
    // 每层的配置
    const KVLayerSpec layer_spec{
        .n_kv_heads = *n_kv_heads,
        .k_head_dim = k_head_dim,
        .v_head_dim = v_head_dim,
        .k_type = GGML_TYPE_F32,
        .v_type = GGML_TYPE_F32,
    };
    std::vector<KVLayerSpec> kv_layers(*n_layers, layer_spec);
    // 当前单 GPU 模式：权重很小，都在gpu，这里先固定gpu了，后期要根据model里面的权重分配情况
    std::vector<Backend *> kv_backends(*n_layers, gpu_backend.get());
    KVCacheConfig kv_config{
        .layers = *n_layers,
        .max_tokens_per_sequence = context_limit,
        .block_tokens = 0,
    };
    KVCacheManager kv_manager(std::make_unique<ContiguousKVStrategy>(std::make_unique<NaiveKVStorage>()));
    if (auto r = kv_manager.init(kv_layers, kv_backends, kv_config); !r){
        std::cerr << "KV 缓存初始化失败：" << r.error().message() << '\n';
        return 1;
    }
    // 创建一个会话的id，一会kvcache会跟这个id 对应
    auto id_result = kv_manager.create_sequence();
    if(!id_result){
        std::cerr << "KV 缓存 创建id失败：" << id_result.error().message() << '\n';
        return 1;
    }
    auto& id = *id_result;

     // -------------------------------- 初始化调度器 和 采样器---------------------------
    Scheduler sched(devices);
    Sampler sampler;
    std::vector<int32_t> generated;
    // ------------------------------- 建图的上下文 ------------------------------------------
    
    // 初始化图的上下文
    GraphContext graph_ctx(cpu_backend.get());

    // 创建输入的tensor，一会送给图的上下文
    std::vector<int32_t> positions;
    Tensor *input_tokens = graph_ctx.make_tensor("input_tokens", DataType::GGML_TYPE_I32, {1});
    input_tokens->tensorRole = TensorRole::TNESOR_ROLE_INPUT;
    input_tokens->backend = cpu_backend.get();
    input_tokens->device = DeviceType::CPU;
    Tensor *input_pos = graph_ctx.make_tensor("input_pos", DataType::GGML_TYPE_I32, {1});
    input_pos->tensorRole = TensorRole::TNESOR_ROLE_INPUT;
    input_pos->backend = cpu_backend.get();
    input_pos->device = DeviceType::CPU;
    graph_ctx.set_input_tokens(input_tokens);
    graph_ctx.set_input_pos(input_pos);

    // bind kvcache manager
    if (auto r = graph_ctx.set_kv_cache(&kv_manager, id); !r){
        std::cerr << "绑定 KV 失败：" << r.error().message() << '\n';
        return 1;
    }
    const size_t max_new_tokens = 8;
    bool failed = false;
    bool pending = false;
    for (size_t step = 0; step < max_new_tokens; ++step){
        auto length = kv_manager.length(id);
        if (!length){
            std::cerr << "查询 KV 长度失败：" << length.error().message() << '\n';
            failed = true;
            break;
        }
        if (*length > context_limit || token_ids.size() > context_limit - *length){
            break;
        }
        // 准备位置
        const int64_t n = static_cast<int64_t>(token_ids.size());
        const int64_t bytes = n * sizeof(int32_t);

        // 首轮 Prefill 从位置 0 开始；后续 Decode 从已提交的 KV 长度开始
        const int64_t n_past = static_cast<int64_t>(*length);
        positions.resize(n);
        for (int64_t i = 0; i < n; ++i){
            positions[i] = static_cast<int32_t>(n_past + i);
        }

        // 更新形状和字节步长
        for (Tensor *t : {input_tokens, input_pos}){
            t->dims = {n, 1, 1, 1};
            t->strides = {sizeof(int32_t), bytes, bytes, bytes};
        }

        // 借用 vector 的内存，不复制
        input_tokens->data = token_ids.data();
        input_pos->data = positions.data();

        // 预留本轮的大小，也就是本轮prefill 有多少个token要加进去
        if (auto r = kv_manager.reserve_append(id, token_ids.size()); !r){
            std::cerr << "预留 KV 失败：" << r.error().message() << '\n';
            failed = true;
            break;
        }

        pending = true;

        // ----------------------------------- 建qwen3的图 --------------------------
        auto result_graph=qwen3->build_graph(graph_ctx);
        if(!result_graph){
            std::cerr << "建图失败" << result_graph.error().message() << '\n';
            failed = true;
            break;
        }
        auto& graph_qwen3 = *result_graph;

        if (auto r = graph_qwen3.compute(); !r){
            std::cerr << "图准备失败：" << r.error().message() << '\n';
            // 此处也需要归还建图阶段取得的 KV 访问
            failed = true;
            break;
        }
   

        // 执行并等待 GPU 完成；失败时也先等待，再归还 KV 访问。
        auto execution = sched.execute(graph_qwen3);
        const cudaError_t sync_error = cudaDeviceSynchronize();
        if (!execution || sync_error != cudaSuccess){
            std::cerr << "执行失败："
                      << (!execution ? execution.error().message() : cudaGetErrorString(sync_error)) << '\n';
            failed = true;
            break;
        }
        if (auto r = graph_ctx.release_kv_access(); !r){
            std::cerr << "归还 KV 失败：" << r.error().message() << '\n';
            failed = true;
            break;
        }
        if (auto r = kv_manager.commit_append(id); !r){
            std::cerr << "提交 KV 失败：" << r.error().message() << '\n';
            failed = true;
            break;
        }
        pending = false;

        // 采样
        auto logits = read_logits(*graph_qwen3.outputs().front(),token_ids.size() - 1); // 取本轮最后一个位置

        if (!logits){
            std::cerr << "读取 logits 失败：" << logits.error().message() << '\n';
            failed = true;
            break;
        }

   
        auto sampled = sampler.sample(*logits);
        if (!sampled){
            std::cerr << "采样失败：" << sampled.error().message() << '\n';
            failed = true;
            break;
        }


        if (eos_token && static_cast<uint32_t>(*sampled) == *eos_token){
            break;
        }
        generated.push_back(*sampled);
        token_ids.assign(1, *sampled); // 下一轮只输入刚生成的一个 token
    }

    // 正常结束与失败退出都归还访问；只有未提交的追加才取消。
    if (auto r = graph_ctx.release_kv_access(); !r){
        std::cerr << "归还 KV 失败：" << r.error().message() << '\n';
        return 1;
    }
    if (pending){
        if (auto r = kv_manager.abort_append(id); !r){
            std::cerr << "取消 KV 失败：" << r.error().message() << '\n';
            return 1;
        }
    }
    if (auto r = kv_manager.release_sequence(id); !r){
        std::cerr << "释放 KV 序列失败：" << r.error().message() << '\n';
        return 1;
    }
    if (failed) return 1;

    // 采样完成之后交给tokenizer - > decode
    auto text = tokenizer->decode(generated);
    if (!text){
        std::cerr << "解码失败：" << text.error().message() << '\n';
        return 1;
    }
    std::cout << *text << std::flush;

    return 0;
}