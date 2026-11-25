// Reuse the benchmark's engine, KV transactions, and experimental Quest build.
#define INFERLLM_QUEST_NO_MAIN
#include "../../sparse-attention-accuracy/quest/run.cpp"

namespace CUDA_F32 {
void quest_selection_trace(std::FILE*, int, const char*);
}

int main(int argc, char** argv) try {
    if (argc != 5 && argc != 6) throw std::runtime_error("usage: replay_worker MODEL QUEST_TRACE BLOCK BUDGET [CONTEXT]");
    const int block = std::stoi(argv[3]), budget = std::stoi(argv[4]);
    if (block <= 0 || budget <= 0 || budget % block) throw std::runtime_error("invalid block budget");
    std::ostream protocol(std::cout.rdbuf());
    QuietOutput quiet;
    auto gguf = take(GGUFContext::load(argv[1]), "GGUF");
    const auto& metadata = gguf.metadata();
    auto tokenizer = take(Tokenizer::from_gguf(metadata), "tokenizer");
    const auto arch = take(metadata.get_meta<std::string>("general.architecture"), "architecture");
    auto meta = [&](const char* name) {
        return take(metadata.get_meta<uint32_t>(arch + "." + name), name);
    };
    const auto model_capacity = meta("context_length");
    const size_t capacity = argc == 6 ? std::stoul(argv[5]) : std::min<uint32_t>(8192, model_capacity);
    if (!capacity || capacity > model_capacity)
        throw std::runtime_error("context must fit the model context length");
    std::vector<KVLayerSpec> layers(meta("block_count"), {
        meta("attention.head_count_kv"), meta("attention.key_length"), meta("attention.value_length"),
        GGML_TYPE_F32, GGML_TYPE_F32});
    register_all_backends();
    auto& registry = BackendRegistry::instance();
    registry.probe_all();
    auto cpu = take(registry.create("cpu", PoolConfig{}), "CPU");
    auto gpu = take(registry.create("cuda:0", PoolConfig{}), "CUDA");
    register_all_models();
    auto model = take(ModelBase::load_model(std::move(gguf)), "model");
    check(model->load_hparams(), "hparams");
    check(model->load_weight(), "weights");
    check(model->plan_weights_placement({gpu.get(), cpu.get()},
          PlacementConfig{.n_ctx = static_cast<uint32_t>(capacity)}), "placement");
    Runner runner(*model, *cpu, gpu.get(), layers, capacity);
    auto trace = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(argv[2], "w"), &std::fclose);
    if (!trace) throw std::runtime_error("cannot open Quest trace");
    runner.begin_request();
    size_t position = 0;
    for (int call = 1; ; ++call) {
        std::string header;
        if (!std::getline(std::cin, header)) break;
        size_t bytes = 0, reply_bytes = 0;
        if (!(std::istringstream(header) >> bytes >> reply_bytes) || !bytes || !reply_bytes)
            throw std::runtime_error("invalid replay frame");
        auto read_tokens = [&](size_t length) {
            std::string text(length, '\0');
            if (!std::cin.read(text.data(), static_cast<std::streamsize>(length)))
                throw std::runtime_error("truncated replay frame");
            return take(tokenizer.encode(text, {.parse_special=true, .add_bos=false, .add_eos=false}), "encode");
        };
        auto input = read_tokens(bytes);
        auto reply = read_tokens(reply_bytes);
        if (input.empty() || reply.empty()) throw std::runtime_error("empty replay segment");
        if (position + input.size() + reply.size() > capacity) {
            protocol << "context_limit " << position << ' ' << input.size() << ' '
                     << reply.size() << '\n' << std::flush;
            break;
        }

        CUDA_F32::quest_experiment_configure(nullptr, nullptr, 0, 0);
        for (size_t start = 0; start + 1 < input.size();) {
            const auto end = std::min(start + 128, input.size() - 1);
            runner.evaluate({input.begin() + start, input.begin() + end}, false);
            start = end;
        }
        CUDA_F32::quest_experiment_configure(nullptr, nullptr, block, budget);
        CUDA_F32::quest_selection_trace(trace.get(), call, "boundary");
        runner.evaluate({input.back()}, false);
        CUDA_F32::quest_selection_trace(trace.get(), call, "decode");
        for (const auto token : reply) runner.evaluate({token}, false);
        CUDA_F32::quest_selection_trace(nullptr, 0, nullptr);
        if (std::fflush(trace.get()) != 0) throw std::runtime_error("flush failed");
        protocol << "ok " << position << ' ' << input.size() << ' ' << reply.size()
                 << '\n' << std::flush;
        position += input.size() + reply.size();
    }
    runner.end_request();
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
