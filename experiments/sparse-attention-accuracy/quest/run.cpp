// Reuse the CUDA benchmark's model execution and KV transaction handling.
#define main cuda_benchmark_main
#include "../../../benchmarks/cuda/bench.cpp"
#undef main
#include "tokenizer.hpp"
#include <cstdio>
namespace CUDA_F32 {
void quest_experiment_configure(std::FILE*, const char*, int, int);
}

// The sweep uses ContiguousKVStrategy and NaiveKVStorage. They do not
// modify committed history on append, so each variant can overwrite its tail.
struct QuestSweepAccess {
    static void rewind(Runner& runner, size_t checkpoint) {
        auto& state = runner.cache_._sequences.at(runner.sequence_);
        if (runner.pending_ || state.pending || checkpoint > state.committed_length)
            throw std::logic_error("invalid document checkpoint");
        check(runner.context_.release_kv_access(), "release before rewind");
        state.committed_length = checkpoint;
    }
};

int main(int argc, char** argv) try {
    const bool inventory = argc >= 3 && std::string_view(argv[2]) == "inventory";
    if (argc < (inventory ? 4 : 7))
        throw std::runtime_error("usage: run MODEL inventory PROMPT... | run MODEL OUTPUT_DIR trace|quality|timing|sweep BLOCK BUDGET PROMPT...");
    std::ostream report(std::cout.rdbuf());
    QuietOutput quiet;
    auto gguf = take(GGUFContext::load(argv[1]), "GGUF");
    const auto& metadata = gguf.metadata();
    auto tokenizer = take(Tokenizer::from_gguf(metadata), "tokenizer");
    auto arch = take(metadata.get_meta<std::string>("general.architecture"), "architecture");
    auto meta = [&](const char* key) {
        return take(metadata.get_meta<uint32_t>(arch + "." + key), key);
    };
    constexpr size_t max_answer = 24;
    const auto model_context = meta("context_length");
    if (model_context <= max_answer) throw std::runtime_error("model context too small");
    const size_t max_prompt = std::getenv("QUEST_MAX_PROMPT")
        ? number(std::getenv("QUEST_MAX_PROMPT"), false) : model_context - max_answer;
    if (max_prompt > model_context - max_answer)
        throw std::runtime_error("QUEST_MAX_PROMPT exceeds model context minus answer reserve");
    struct Prompt {
        std::filesystem::path path;
        std::vector<int32_t> ids;
        size_t split;
    };
    std::vector<Prompt> prompts;
    size_t longest = 0;
    if (inventory) report << "sample,prompt_tokens,document_tokens,model_context\n";
    for (int i = inventory ? 3 : 6; i < argc; ++i) {
        const std::filesystem::path path(argv[i]);
        std::ifstream stream(path);
        if (!stream) throw std::runtime_error("cannot read prompt: " + path.string());
        std::string prompt((std::istreambuf_iterator<char>(stream)), {});
        const auto boundary = prompt.rfind("\n\nNow, answer the following question");
        if (boundary == std::string::npos) throw std::runtime_error("question boundary missing");
        auto ids = take(tokenizer.encode(prompt, {.parse_special=true}), "encode");
        auto prefix = take(tokenizer.encode(std::string_view(prompt).substr(0, boundary),
                                             {.parse_special=true}), "encode document");
        const auto common = std::mismatch(prefix.begin(), prefix.end(), ids.begin(), ids.end());
        const size_t split = common.second - ids.begin();
        if (split == 0 || split >= ids.size())
            throw std::runtime_error("invalid prompt boundary: " + path.string());
        if (inventory) {
            report << path.stem().string() << ',' << ids.size() << ',' << split << ',' << model_context << '\n';
            continue;
        }
        if (ids.size() > max_prompt)
            throw std::runtime_error("prompt exceeds limit: " + path.string() + " (" +
                                     std::to_string(ids.size()) + " tokens)");
        longest = std::max(longest, ids.size());
        prompts.push_back({path, std::move(ids), split});
    }
    if (inventory) return 0;

    register_all_backends();
    auto& registry = BackendRegistry::instance();
    registry.probe_all();
    auto cpu = take(registry.create("cpu", PoolConfig{}), "CPU");
    auto gpu = take(registry.create("cuda:0", PoolConfig{}), "CUDA");
    std::vector<KVLayerSpec> layers(meta("block_count"), {
        meta("attention.head_count_kv"), meta("attention.key_length"), meta("attention.value_length"),
        GGML_TYPE_F32, GGML_TYPE_F32});
    register_all_models();
    auto model = take(ModelBase::load_model(std::move(gguf)), "model");
    check(model->load_hparams(), "hparams");
    check(model->load_weight(), "weights");
    PlacementConfig placement;
    placement.n_ctx = static_cast<uint32_t>(longest + max_answer);
    std::vector<Backend*> devices{gpu.get(), cpu.get()};
    check(model->plan_weights_placement(devices, placement), "placement");
    Runner runner(*model, *cpu, gpu.get(), layers, placement.n_ctx);
    const auto eos = tokenizer.eos_id();
    const std::string mode(argv[3]);
    const int block = std::stoi(argv[4]);
    const int budget = std::stoi(argv[5]);
    if ((mode != "trace" && mode != "quality" && mode != "timing" && mode != "sweep") ||
        (mode != "trace" && (block <= 0 || budget <= 0 || budget % block)) ||
        (mode == "quality" && (budget / block < 2 || (budget / block) % 2)))
        throw std::runtime_error("invalid mode or Quest configuration");
    for (const auto& [path, ids, split] : prompts) {
        // Limit the quadratic score workspace as the prefix grows.
        const size_t chunk = ids.size() <= 4096 ? 512 : ids.size() <= 8192 ? 256 : 128;
        const auto stem = std::filesystem::path(argv[2]) / path.stem();
        std::unique_ptr<std::FILE, decltype(&std::fclose)> trace(nullptr, &std::fclose);
        if (mode == "trace") {
            trace.reset(std::fopen((stem.string() + ".csv").c_str(), "w"));
            if (!trace) throw std::runtime_error("cannot open trace");
            std::fprintf(trace.get(), "phase,length,layer,head,method,block,budget,read,mass\n");
        }
        const bool sweep = mode == "sweep";
        if (sweep && ((block != 16 && block != 32 && block != 64) ||
                      (budget != 256 && budget != 512 && budget != 1024 && budget != 2048)))
            throw std::runtime_error("sweep expects block size 16, 32 or 64 and a budget from the scan");
        const auto budgets = sweep
            ? (std::getenv("QUEST_VALIDATION") ? std::vector{0, budget}
                                               : std::vector{0, 256, 512, 1024, 2048})
            : mode == "trace" ? std::vector{0} : mode == "timing"
            ? std::vector{0, budget} : std::vector{0, budget, budget / 2};
        bool document_ready = false;
        for (int selected_budget : budgets) {
            const std::string label = selected_budget == 0 ? "dense" :
                sweep ? "b" + std::to_string(selected_budget) :
                selected_budget == budget ? "quest" : "quest_half";
            const std::filesystem::path output_path(stem.string() + "." + label);
            if (sweep && std::filesystem::exists(output_path)) continue;
            auto configure = [&](const char* phase) {
                CUDA_F32::quest_experiment_configure(trace.get(), phase, selected_budget ? block : 0, selected_budget);
            };
            const auto started = Clock::now();
            configure(nullptr);
            if (sweep && document_ready) {
                QuestSweepAccess::rewind(runner, split);
            } else {
                runner.begin_request();
                for (size_t start = 0; start < split; start += chunk) {
                    const auto end = std::min(start + chunk, split);
                    runner.evaluate({ids.begin()+start, ids.begin()+end}, false);
                }
                document_ready = sweep;
            }
            const auto prefilled = Clock::now();
            int32_t next = -1;
            for (size_t pos = split; pos < ids.size(); ++pos) {
                configure("question");
                next = runner.evaluate({ids[pos]}, pos + 1 == ids.size()).token;
            }
            const auto questioned = Clock::now();
            std::vector<int32_t> generated;
            while (generated.size() < max_answer && (!eos || next != *eos)) {
                generated.push_back(next);
                if (generated.size() == max_answer) break;
                configure("answer");
                next = runner.evaluate({next}, true).token;
            }
            const auto answered = Clock::now();
            configure(nullptr);
            if (!sweep) runner.end_request();
            auto temporary = output_path;
            temporary += ".tmp";
            std::ofstream output(temporary);
            output << take(tokenizer.decode(generated), "decode");
            output.close();
            if (!output) throw std::runtime_error("cannot write answer");
            std::filesystem::rename(temporary, output_path);
            report << path.stem() << ' ' << ids.size() << ' ' << label << ' '
                   << generated.size() << ' ' << seconds(started, prefilled) << ' '
                   << seconds(prefilled, questioned) << ' ' << seconds(questioned, answered)
                   << std::endl;
        }
        if (document_ready) runner.end_request();
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
