#include <iostream>
#include <memory>
#include<../include/backend.hpp>
#include<../include/gguf_parser.hpp>
#include <../include/model.hpp>
#include<../include/graph.hpp>
#include <../include/scheduler.hpp>
#include "sampler.hpp"
#include "tokenizer.hpp"
#include <charconv>
#include <limits>
#include <numeric>
#include <optional>

int main(int argc, char** argv){
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
    // 实例化设备 这里选择cpu先，先把cpu跑起来
    PoolConfig pool_cfg = { 
                            .weights = Strategy::NAIVE,
                            .activation = Strategy::NAIVE,
                            .kvcache = Strategy::NAIVE,
    };
    std::vector<Backend*> bc;
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

    // ---------------加载backend完成，并且只是指定了cpu-------------
    //     name:         cuda:0
    // DeviceType:   CUDA
    // memory_bytes: 7441743872
    // name:         cpu
    // DeviceType:   CPU
    // memory_bytes: 5772193792
    // cpu 实例化中 device_id: 0
    // [cpu] default threads = 8
    // cpu 析构中 device_id: 0


    // ---------------gguf的解析-----------------
    auto gguf=GGUFContext::load("/home/wcf/Infer-LLM/models/Qwen3-0.6B-F32.gguf");
    if (!gguf) {
        std::cerr << "加载 GGUF 失败：" << gguf.error().message() << '\n';
        return 1;
    }
    (*gguf).print(10);                                                              // 打印10个tensor
    // 在 GGUF 移入 Model 前读取本示例的生成边界，不把模型信息放进 Sampler。
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
    // 放置权重
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
    std::vector<int32_t> positions;
    // context 跨轮复用 RoPE 表；输入数据仍归上面的 vector 所有。
    GraphContext graph_ctx;
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

    Sampler sampler;
    // Scheduler 跨轮保留 Activation Buffer；析构时统一释放，且早于 Backend 析构。
    Scheduler sched(bc);
    std::vector<int32_t> generated;
    std::cout << "Greedy 生成，最多 " << max_new_tokens << " 个新 token；无 KV cache，每轮重算完整前缀\n";
    for (size_t step = 0; step < max_new_tokens; ++step) {
        // 保证本轮输入和新 token 的位置均不超过模型上下文。
        if (token_ids.size() >= *context_length ||
            token_ids.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            std::cout << "达到上下文长度限制，停止生成\n";
            break;
        }
        positions.resize(token_ids.size());
        std::iota(positions.begin(), positions.end(), int32_t{0});
        const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
        const int64_t input_bytes = n_tokens * sizeof(int32_t);
        for (Tensor* input : {input_tokens, input_pos}) {
            input->dims = {n_tokens, 1, 1, 1};
            input->strides = {sizeof(int32_t), input_bytes, input_bytes, input_bytes};
        }
        input_tokens->data = token_ids.data();
        input_pos->data = positions.data();

        int32_t next_token;
        {
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
            if (auto r = sched.execute(g); !r) {
                std::cerr << "执行失败：" << r.error().message() << '\n';
                return 1;
            }
            if (g.outputs().size() != 1 || !g.outputs().front()) {
                std::cerr << "预期一个 logits 输出\n";
                return 1;
            }
            auto logits = read_logits(*g.outputs().front(), token_ids.size() - 1);
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
            std::cout << "[step " << step + 1 << "] context=" << n_tokens
                      << " token_id=" << next_token
                      << " logit=" << (*logits)[next_token] << std::endl;
        }
        if (eos_token && next_token == *eos_token) {
            std::cout << "遇到 EOS，停止生成\n";
            break;
        }
        // 图已释放，更新 vector 不会使正在执行的图持有悬空输入。
        generated.push_back(next_token);
        token_ids.push_back(next_token);
    }
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
