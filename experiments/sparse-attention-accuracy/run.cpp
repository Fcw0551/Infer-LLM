// Reuse the CUDA benchmark's model execution and KV transaction handling.
#define main cuda_benchmark_main
#include "../../benchmarks/cuda/bench.cpp"
#undef main
#include "tokenizer.hpp"

int main(int argc, char** argv) try {
    if (argc < 4) throw std::runtime_error("usage: run MODEL OUTPUT_DIR PROMPT...");
    std::ostream report(std::cout.rdbuf());
    QuietOutput quiet;
    register_all_backends();
    auto& registry = BackendRegistry::instance();
    registry.probe_all();
    auto cpu = take(registry.create("cpu", PoolConfig{}), "CPU");
    auto gpu = take(registry.create("cuda:0", PoolConfig{}), "CUDA");
    auto gguf = take(GGUFContext::load(argv[1]), "GGUF");
    const auto& metadata = gguf.metadata();
    auto tokenizer = take(Tokenizer::from_gguf(metadata), "tokenizer");
    auto arch = take(metadata.get_meta<std::string>("general.architecture"), "architecture");
    auto meta = [&](const char* key) {
        return take(metadata.get_meta<uint32_t>(arch + "." + key), key);
    };
    std::vector<KVLayerSpec> layers(meta("block_count"), {
        meta("attention.head_count_kv"), meta("attention.key_length"), meta("attention.value_length"),
        GGML_TYPE_F32, GGML_TYPE_F32});
    register_all_models();
    auto model = take(ModelBase::load_model(std::move(gguf)), "model");
    check(model->load_hparams(), "hparams");
    check(model->load_weight(), "weights");
    constexpr size_t max_prompt = 4096, max_answer = 24, chunk = 512;
    PlacementConfig placement;
    placement.n_ctx = max_prompt + max_answer;
    std::vector<Backend*> devices{gpu.get(), cpu.get()};
    check(model->plan_weights_placement(devices, placement), "placement");
    Runner runner(*model, *cpu, gpu.get(), layers, placement.n_ctx);
    const auto eos = tokenizer.eos_id();
    struct Mode { const char* name; uint32_t window, sink; };
    const Mode modes[] = {{"dense", 0, 0}, {"recent512", 512, 0}, {"sink4_recent508", 508, 4}};
    int remaining = 6;
    for (int i = 3; i < argc; ++i) {
        const std::filesystem::path path(argv[i]);
        std::ifstream stream(path);
        if (!stream) throw std::runtime_error("cannot read prompt");
        std::string prompt((std::istreambuf_iterator<char>(stream)), {});
        auto ids = take(tokenizer.encode(prompt, {.parse_special=true}), "encode");
        if (ids.size() < 1024 || ids.size() > max_prompt) {
            report << path.stem() << " skip " << ids.size() << std::endl;
            continue;
        }
        for (const auto& mode : modes) {
            runner.begin_request();
            int32_t next = -1;
            for (size_t start = 0; start < ids.size(); start += chunk) {
                const auto end = std::min(start + chunk, ids.size());
                next = runner.evaluate({ids.begin()+start, ids.begin()+end}, end == ids.size()).token;
            }
            std::vector<int32_t> generated;
            while (generated.size() < max_answer && (!eos || next != *eos)) {
                generated.push_back(next);
                if (generated.size() == max_answer) break;
                next = runner.evaluate({next}, true, mode.window, mode.sink).token;
            }
            auto answer = take(tokenizer.decode(generated), "decode");
            runner.end_request();
            std::ofstream output(std::filesystem::path(argv[2]) / (path.stem().string() + "." + mode.name));
            output << answer;
            if (!output) throw std::runtime_error("cannot write answer");
            report << path.stem() << ' ' << ids.size() << ' ' << mode.name << ' ' << generated.size() << std::endl;
        }
        if (--remaining == 0) break;
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
