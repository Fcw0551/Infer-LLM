#include <iostream>
#include <cuda_runtime_api.h>
#include <memory>
#include<../include/backend.hpp>
#include<../include/gguf_parser.hpp>
#include <../include/model.hpp>
#include<../include/graph.hpp>
#include <../include/scheduler.hpp>
#include "sampler.hpp"
#include "tokenizer.hpp"
#include "kvcache/kvcacheManager.hpp"
#include "kvcache/KVStorages/naiveKVStorage.hpp"
#include "kvcache/KVStrategys/contiguousKVStrategy.hpp"
#include <algorithm>
#include <charconv>
#include <limits>
#include <numeric>
#include <optional>

int main(int argc, char** argv){
    // ----- 解析生成参数 ------
    size_t max_new_tokens = 8;
    if (argc > 2) {
        std::cerr << "用法：inferllm_main [max_new_tokens]\n";
        return 1;
    }
    if (argc == 2) {
        const std::string_view arg(argv[1]);
        const auto [end, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), max_new_tokens);
        if (ec != std::errc{} || end != arg.data() + arg.size() || max_new_tokens == 0) {
            std::cerr << "max_new_tokens 必须为正整数\n";
            return 1;
        }
    }
    // ----------- 加载backend ------------
    // 注册 cpu gpu等
    register_all_backends();
    // 单例注册器
    BackendRegistry& reg = BackendRegistry::instance();
    // 探测
    reg.probe_all();
    // 打印
    reg.print();
    // CUDA 优先；保留 CPU 用于输入、embedding 和权重放置兜底
    PoolConfig pool_cfg = { 
                            .weights = Strategy::NAIVE,
                            .activation = Strategy::NAIVE,
                            .kvcache = Strategy::NAIVE,
    };
    std::vector<Backend*> bc;
    auto gpu_result = reg.create("cuda:0", pool_cfg);
    if (!gpu_result) {
        std::cerr << "创建 CUDA 后端失败：" << gpu_result.error().message() << '\n';
        return 1;
    }
    std::unique_ptr<Backend> gpu_backend = std::move(*gpu_result);
    bc.push_back(gpu_backend.get());
    auto result=reg.create("cpu",pool_cfg);
    if(!result){
        const auto &ec = result.error();
        std::cerr << "创建后端失败: " << ec.message()
                  << "，错误码: " << ec.value()
                  << "，类别: " << ec.category().name()
                  << '\n';
        return 1;
    }
    std::unique_ptr<Backend> cpu_backend = std::move(*result);
    bc.push_back(cpu_backend.get());

    std::cout << "推理设备优先级：cuda:0 → cpu\n";

    // ---------------gguf的解析-----------------
    auto gguf=GGUFContext::load("/home/wcf/Infer-LLM/models/Qwen3-0.6B-F32.gguf");
    if (!gguf) {
        std::cerr << "加载 GGUF 失败：" << gguf.error().message() << '\n';
        return 1;
    }
    (*gguf).print(10);                                                              // 打印10个tensor
    // ----- 读取上下文长度、词表和停止条件 ------
    // 在 GGUF 移入 Model 前读取生成边界。
    const auto arch = gguf->metadata().get_meta<std::string>("general.architecture");
    if (!arch) {
        std::cerr << "缺少模型架构信息\n";
        return 1;
    }
    const auto context_length = gguf->metadata().get_meta<uint32_t>(*arch + ".context_length");
    const auto vocab_size = gguf->metadata().get_meta_array_len("tokenizer.ggml.tokens");
    if (!context_length || *context_length == 0 || !vocab_size || *vocab_size == 0) {
        std::cerr << "上下文长度或词表信息无效\n";
        return 1;
    }
    std::optional<int32_t> eos_token;
    if (auto eos = gguf->metadata().get_meta<uint32_t>("tokenizer.ggml.eos_token_id"); eos) {
        if (*eos >= *vocab_size || *eos > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            std::cerr << "EOS token ID 无效\n";
            return 1;
        }
        eos_token = static_cast<int32_t>(*eos);
    }
    // ----- 读取 KV 缓存需要的层数、头数和维度 ------
    // 在 GGUF 移入 Model 前保存；与 load_hparams 的默认维度规则一致
    const auto n_layers = gguf->metadata().get_meta<uint32_t>(*arch + ".block_count");
    const auto n_embd = gguf->metadata().get_meta<uint32_t>(*arch + ".embedding_length");
    const auto n_heads = gguf->metadata().get_meta<uint32_t>(*arch + ".attention.head_count");
    const auto n_kv_heads = gguf->metadata().get_meta<uint32_t>(*arch + ".attention.head_count_kv");
    if (!n_layers || !n_embd || !n_heads || !n_kv_heads ||
        *n_layers == 0 || *n_embd == 0 || *n_heads == 0 || *n_kv_heads == 0) {
        std::cerr << "KV 缓存模型参数无效\n";
        return 1;
    }
    const auto key_length = gguf->metadata().get_meta<uint32_t>(*arch + ".attention.key_length");
    const auto value_length = gguf->metadata().get_meta<uint32_t>(*arch + ".attention.value_length");
    const uint32_t k_head_dim = key_length.value_or(*n_embd / *n_heads);
    const uint32_t v_head_dim = value_length.value_or(*n_embd / *n_heads);
    if (k_head_dim == 0 || v_head_dim == 0) {
        std::cerr << "KV head 维度无效\n";
        return 1;
    }

    // ----- 初始化 tokenizer ------
    auto tokenizer = Tokenizer::from_gguf(gguf->metadata());
    if (!tokenizer) {
        std::cerr << "加载 tokenizer 失败：" << tokenizer.error().message() << '\n';
        return 1;
    }
    // -------------加载模型--------------
    register_all_models();
    // -------------传ggufcontext给modelbase，进而实例化出模型----------------
    auto model_result = ModelBase::load_model(std::move(*gguf));
    if (!model_result)
    {
        std::cerr << "实例化模型失败: "
                  << model_result.error().message() << '\n';
        return 1;
    }
    // --------------实例化出模型-----------------
    std::unique_ptr<ModelBase> qwen3_ptr = std::move(*model_result);
    if (auto r = qwen3_ptr->load_hparams(); !r) {
        std::cerr << "加载超参数失败：" << r.error().message() << '\n';
        return 1;
    }
    auto load_weight_result=qwen3_ptr->load_weight();
    if (!load_weight_result){
        const auto &ec = load_weight_result.error();
        std::cerr << "load_weight失败: " << ec.message()
                  << "，错误码: " << ec.value()
                  << "，类别: " << ec.category().name()
                  << '\n';
        return 1;
    }
    // ----- 放置模型权重 ------
    PlacementConfig placement_cfg{ 1024, 4, 1ull << 30, 256ull << 20};
    auto placement_result = qwen3_ptr->plan_weights_placement(bc, placement_cfg);

    if (!placement_result){
        std::cerr << "权重放置失败: "
                  << placement_result.error().message() << '\n';
        return 1;
    }

    // ------------将 prompt 编码为输入 token ID--------------
    // Qwen3 单轮对话模板，使用空思考块关闭思考模式。
    const std::string_view prompt =
        "<|im_start|>user\n"
        "介绍一下你自己<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n";
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
    for (int32_t id : token_ids) {
        if (id < 0 || static_cast<uint64_t>(id) >= *vocab_size) {
            std::cerr << "输入 token ID 超出词表\n";
            return 1;
        }
    }
    // ----- 初始化 KV 缓存管理器 ------
    const size_t context_limit = std::min<size_t>(1024, std::numeric_limits<int32_t>::max());
    if (token_ids.size() > context_limit) {
        std::cerr << "prompt 超过上下文长度限制\n";
        return 1;
    }
    // 最后一次采样的 token 不再送回模型，因此最多追加 max_new_tokens - 1 个 KV。
    const size_t cache_capacity = token_ids.size() + std::min(max_new_tokens - 1, context_limit - token_ids.size());
    const KVLayerSpec layer_spec{
        .n_kv_heads = *n_kv_heads,
        .k_head_dim = k_head_dim,
        .v_head_dim = v_head_dim,
        .k_type = GGML_TYPE_F32,
        .v_type = GGML_TYPE_F32,
    };
    std::vector<KVLayerSpec> kv_layers(*n_layers, layer_spec);
    // 当前单 GPU 模式：全部 KV 常驻 cuda:0，跨设备输入由调度器复制。
    std::vector<Backend*> kv_backends(*n_layers, gpu_backend.get());
    KVCacheConfig kv_config{
        .layers = *n_layers,
        .max_tokens_per_sequence = cache_capacity,
        .block_tokens = 0,
    };
    KVCacheManager kv_manager(std::make_unique<ContiguousKVStrategy>(std::make_unique<NaiveKVStorage>()));
    if (auto r = kv_manager.init(kv_layers, kv_backends, kv_config); !r) {
        std::cerr << "KV 缓存初始化失败：" << r.error().message() << '\n';
        return 1;
    }

    // ----- 创建跨轮复用的计算图上下文和输入张量 ------
    std::vector<int32_t> positions;
    // context 跨轮复用 RoPE 表；输入数据仍归上面的 vector 所有。
    GraphContext graph_ctx(cpu_backend.get());
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

    // ----- 创建序列并将缓存借给图上下文 ------
    auto sequence = kv_manager.create_sequence();
    if (!sequence) {
        std::cerr << "创建 KV 序列失败：" << sequence.error().message() << '\n';
        return 1;
    }
    const SequenceId sequence_id = *sequence;
    // TODO：由 GraphContext 补齐以下两个接口：
    // std::expected<void, std::error_code> set_kv_cache(KVCacheManager* manager, SequenceId id) noexcept;
    // std::expected<void, std::error_code> release_kv_access() noexcept;
    // 前者只借用 manager、保存 id；建图时获取的各层 access 由 context 保存。
    // 后者逐个 release_layer 并清空访问列表，空列表重复调用应成功
    // ----- 统一处理错误退出时的访问归还和序列释放 ------
    // 每轮在归还 KV 访问前等待 GPU 完成，包括执行失败的路径。
    struct KVRequestScope {
        KVCacheManager& manager;
        GraphContext& context;
        SequenceId id;
        bool pending = false;
        ~KVRequestScope() {
            if (id == kInvalidSequenceId) return;
            if (auto r = context.release_kv_access(); !r)
                std::cerr << "归还 KV 访问失败：" << r.error().message() << '\n';
            if (pending) {
                if (auto r = manager.abort_append(id); !r)
                    std::cerr << "取消 KV 追加失败：" << r.error().message() << '\n';
            }
            if (id != kInvalidSequenceId) {
                if (auto r = manager.release_sequence(id); !r)
                    std::cerr << "释放 KV 序列失败：" << r.error().message() << '\n';
            }
        }
    } kv_request{kv_manager, graph_ctx, sequence_id};

    // 设置kv模块
    if (auto r = graph_ctx.set_kv_cache(&kv_manager, sequence_id); !r) {
        std::cerr << "绑定 KV 上下文失败：" << r.error().message() << '\n';
        return 1;
    }

    // ----- 初始化采样器和执行调度器 ------
    Sampler sampler;
    // Scheduler 跨轮保留 Activation Buffer；析构时统一释放，且早于 Backend 析构。
    Scheduler sched(bc);
    std::vector<int32_t> generated;
    std::cout << "Greedy 生成，最多 " << max_new_tokens << " 个新 token；首次 Prefill，随后使用 KV cache 增量 Decode\n";
    // ----- 逐轮推理和生成 ------
    for (size_t step = 0; step < max_new_tokens; ++step) {
        // ----- 查询有效历史并检查本轮容量 ------
        auto cached_length = kv_manager.length(sequence_id);
        if (!cached_length) {
            std::cerr << "查询 KV 长度失败：" << cached_length.error().message() << '\n';
            return 1;
        }
        const size_t n_past = *cached_length;
        // token_ids 首轮是 prompt，之后只保存上一轮生成的一个 token。
        const size_t n_new = token_ids.size();
        if (n_past > cache_capacity || n_new > cache_capacity - n_past) {
            std::cout << "达到上下文长度限制，停止生成\n";
            break;
        }

        // ----- 预留本轮新增 KV 的写入范围 ------
        if (auto r = kv_manager.reserve_append(sequence_id, n_new); !r) {
            std::cerr << "预留 KV 空间失败：" << r.error().message() << '\n';
            return 1;
        }
        kv_request.pending = true;

        // ----- 准备本轮 token 和绝对位置输入 ------
        positions.resize(n_new);
        // 把绝对位置先确定，再上一轮的基础上新增加， n_pase n_pase+1 n_pase+2……
        std::iota(positions.begin(), positions.end(), static_cast<int32_t>(n_past));
        const int64_t n_tokens = static_cast<int64_t>(n_new);
        const int64_t input_bytes = n_tokens * sizeof(int32_t);
        for (Tensor* input : {input_tokens, input_pos}) {
            input->dims = {n_tokens, 1, 1, 1};
            input->strides = {sizeof(int32_t), input_bytes, input_bytes, input_bytes};
        }
        input_tokens->data = token_ids.data();
        input_pos->data = positions.data();

        int32_t next_token;
        {
            // ----- 构建并准备本轮计算图 ------
            // 每轮使用新图；激活内存由 Scheduler 持有，错误退出也由其析构回收。
            auto graph_result = qwen3_ptr->build_graph(graph_ctx);
            if (!graph_result) {
                std::cerr << "建图失败：" << graph_result.error().message() << '\n';
                return 1;
            }
            Graph& g = *graph_result;
            if (auto r = g.compute(); !r) {
                std::cerr << "图准备失败：" << r.error().message() << '\n';
                return 1;
            }
            // ----- 执行本轮计算图 ------
            auto execution_result = sched.execute(g);
            const cudaError_t sync_error = cudaDeviceSynchronize();
            if (!execution_result) {
                std::cerr << "执行失败：" << execution_result.error().message() << '\n';
                return 1;
            }
            if (sync_error != cudaSuccess) {
                std::cerr << "GPU 执行失败：" << cudaGetErrorString(sync_error) << '\n';
                return 1;
            }
            // ----- 归还本轮 KV 访问并提交有效长度 ------
            // 只有全部层的 KV 写入与计算真正完成后才能提交
            if (auto r = graph_ctx.release_kv_access(); !r) {
                std::cerr << "归还 KV 访问失败：" << r.error().message() << '\n';
                return 1;
            }
            if (auto r = kv_manager.commit_append(sequence_id); !r) {
                std::cerr << "提交 KV 追加失败：" << r.error().message() << '\n';
                return 1;
            }
            kv_request.pending = false;

            // ----- 读取末尾 logits 并采样 ------
            if (g.outputs().size() != 1 || !g.outputs().front()) {
                std::cerr << "预期一个 logits 输出\n";
                return 1;
            }
            auto logits = read_logits(*g.outputs().front(), n_new - 1);
            if (!logits) {
                std::cerr << "读取 logits 失败：" << logits.error().message() << '\n';
                return 1;
            }
            auto sampled = sampler.sample(*logits);
            if (!sampled) {
                std::cerr << "采样失败：" << sampled.error().message() << '\n';
                return 1;
            }
            next_token = *sampled;
            std::cout << "[step " << step + 1 << "] context=" << (n_past + n_new)
                      << " token_id=" << next_token
                      << " logit=" << (*logits)[next_token] << std::endl;
        }
        // ----- 检查停止条件并准备下一轮输入 ------
        if (eos_token && next_token == *eos_token) {
            std::cout << "遇到 EOS，停止生成\n";
            break;
        }
        // 图已释放，更新 vector 不会使正在执行的图持有悬空输入。
        generated.push_back(next_token);
        token_ids.assign(1, next_token);
    }
    // ----- 释放本次请求的 KV 序列 ------
    if (auto r = kv_manager.release_sequence(sequence_id); !r) {
        std::cerr << "释放 KV 序列失败：" << r.error().message() << '\n';
        return 1;
    }
    kv_request.id = kInvalidSequenceId;
    if (auto r = graph_ctx.set_kv_cache(nullptr, kInvalidSequenceId); !r) {
        std::cerr << "解绑 KV 上下文失败：" << r.error().message() << '\n';
        return 1;
    }

    // ----- 解码并输出生成结果 ------
    std::cout << "生成的 token IDs：";
    for (int32_t id : generated) std::cout << ' ' << id;
    std::cout << '\n';
    auto text = tokenizer->decode(generated);
    if (!text) {
        std::cerr << "生成结果解码失败：" << text.error().message() << '\n';
        return 1;
    }
    std::cout << "生成的文字：" << *text << '\n';
    return 0;

}
