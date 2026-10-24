#include <iostream>
#include <memory>
#include<../include/backend.hpp>
#include<../include/gguf_parser.hpp>
#include <../include/model.hpp>
#include<../include/graph.hpp>
#include <../include/scheduler.hpp>
int main(){
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
    (*gguf).print(10);                                                              // 打印10个tensor
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
    qwen3_ptr->load_hparams();
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

    // ------------------ 建图 ----------------------
    // 手工指定测试 token ID；位置从 0 开始，对应一次 prefill 输入。
    // 数组先于 context 和 graph 创建，保证图使用期间 data 始终有效。
    std::array<int32_t, 4> token_ids{151643, 100, 200, 300};
    std::array<int32_t, 4> positions{0, 1, 2, 3};
    const int64_t n_tokens = static_cast<int64_t>(token_ids.size());
    // 构建上下文
    GraphContext graph_ctx;
    Tensor *input_tokens = graph_ctx.make_tensor("input_tokens", DataType::GGML_TYPE_I32, {n_tokens, 1, 1, 1}); //输入token
    input_tokens->data = token_ids.data();      
    input_tokens->tensorRole = TensorRole::TNESOR_ROLE_INPUT;
    input_tokens->backend = cpu_backend.get();
    input_tokens->device = DeviceType::CPU;

    Tensor *input_pos = graph_ctx.make_tensor("input_pos", DataType::GGML_TYPE_I32, {n_tokens, 1, 1, 1});       // 输入token的位置
    input_pos->data = positions.data();
    input_pos->tensorRole = TensorRole::TNESOR_ROLE_INPUT;
    input_pos->backend = cpu_backend.get();
    input_pos->device = DeviceType::CPU;

    graph_ctx.set_input_tokens(input_tokens);
    graph_ctx.set_input_pos(input_pos);

    // 传入上下文
    auto graph_result = qwen3_ptr->build_graph(graph_ctx);
    if (!graph_result)
    {
        std::cerr << "建图失败: " << graph_result.error().message() << '\n';
        return 1;
    }
    Graph& g=*graph_result;
    // 生成拓扑序，同时检查叶子节点的数据是否就绪
    auto topo_result = g.compute();
    if (!topo_result)
    {
        std::cerr << "图准备失败: "
                  << topo_result.error().message() << '\n';
        return 1;
    }

    // ----------------- 调度器，传图给调度器---------------
    Scheduler sched(bc);
    auto exe_result = sched.execute(g);
    if (!exe_result){
        std::cerr << "执行失败：" << exe_result.error().message() << '\n';
    }

    return 0;

}