#include "backend.hpp"
#include <cuda_runtime_api.h>
#include "backends/cpu_backend.hpp"
#include "gguf_parser.hpp"
#include "graph.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include "scheduler.hpp"
#include "kvcache/kvcacheManager.hpp"
#include "kvcache/KVStorages/naiveKVStorage.hpp"
#include "kvcache/KVStrategys/contiguousKVStrategy.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}

struct Options {
    std::string model = BENCH_DEFAULT_MODEL;
    std::string mode = "all", csv, profile, backend = "cuda:0";
    size_t prompt = 16, generated = 8, repeats = 3, warmup = 1, threads = 0;
    bool help = false;
};

size_t number(std::string_view text, bool allow_zero) {
    size_t n = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data()+text.size(), n);
    if (error != std::errc{} || end != text.data()+text.size() || (!allow_zero && n == 0) ||
        n > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("无效整数参数："+std::string(text));
    return n;
}

Options parse(int argc, char** argv) {
    Options o;
    bool custom_shape = false;
    for (int i=1; i<argc; ++i) {
        const std::string_view key(argv[i]);
        if (key == "--help" || key == "-h") { o.help = true; return o; }
        if (i+1 == argc) throw std::runtime_error("参数缺少值："+std::string(key));
        const std::string_view value(argv[++i]);
        if (key == "--model" || key == "-m") o.model = value;
        else if (key == "--backend") o.backend = value;
        else if (key == "--profile") o.profile = value;
        else if (key == "--mode") o.mode = value;
        else if (key == "--prompt-tokens" || key == "-p") { o.prompt = number(value, false); custom_shape = true; }
        else if (key == "--gen-tokens" || key == "-n") { o.generated = number(value, false); custom_shape = true; }
        else if (key == "--threads" || key == "-t") o.threads = number(value, true);
        else if (key == "--repetitions" || key == "-r") o.repeats = number(value, false);
        else if (key == "--warmup") o.warmup = number(value, true);
        else if (key == "--csv") o.csv = value;
        else throw std::runtime_error("未知参数："+std::string(key));
    }
    if (o.backend != "cuda:0" && o.backend != "cpu")
        throw std::runtime_error("--backend 必须是 cuda:0 或 cpu");
    if (o.mode != "pp" && o.mode != "pg" && o.mode != "all")
        throw std::runtime_error("--mode 必须是 pp、pg 或 all");
    if (!o.profile.empty()) {
        if (custom_shape) throw std::runtime_error("--profile 不可与 -p/-n 混用");
        o.mode = "pg";
        if (o.profile == "p32") o.prompt = 32;
        else if (o.profile == "p64") o.prompt = 64;
        else if (o.profile == "p128") o.prompt = 128;
        else if (o.profile == "p256") o.prompt = 256;
        else throw std::runtime_error("未知 profile："+o.profile);
        o.generated = 128;
    }
    return o;
}

void help() {
    std::cout << "Infer-LLM CUDA C++ benchmark (KV cache)\n"
              << "  --backend cuda:0|cpu      默认 cuda:0；cpu 用于对照\n"
              << "  -m, --model PATH          模型 GGUF，默认仓库内 Qwen3-0.6B-F32\n"
              << "  --mode pp|pg|all          输入处理 / 连续生成 / 两者，默认 all\n"
              << "  --profile NAME           p32=32/128, p64=64/128, p128=128/128, p256=256/128\n"
              << "  -p, --prompt-tokens N     合成输入 token 数，默认 16\n"
              << "  -n, --gen-tokens N        pg 固定采样次数，默认 8\n"
              << "  -t, --threads N           CPU 线程数，默认 0（后端自动检测）\n"
              << "  -r, --repetitions N       正式重复次数，默认 3\n"
              << "  --warmup N               每种模式预热次数，默认 1\n"
              << "  --csv PATH               写出每轮耗时和吞吐量\n"
              << "pp 不分词、不采样；pg 为固定工作量测试，采样到 EOS 也继续。\n";
}

template<class T> T take(std::expected<T, std::error_code> value, const char* stage) {
    if (!value) throw std::runtime_error(std::string(stage)+": "+value.error().message());
    return std::move(*value);
}
void check(std::expected<void, std::error_code> value, const char* stage) {
    if (!value) throw std::runtime_error(std::string(stage)+": "+value.error().message());
}

// 避免引擎调试输出到终端；保留 stderr 错误。计时不依赖日志。
class QuietOutput {
    class Sink : public std::streambuf {
        int_type overflow(int_type c) override { return traits_type::not_eof(c); }
        std::streamsize xsputn(const char*, std::streamsize n) override { return n; }
    } sink_;
    std::streambuf* old_;
public:
    QuietOutput() : old_(std::cout.rdbuf(&sink_)) {}
    ~QuietOutput() { std::cout.rdbuf(old_); }
};

struct Step {
    double prepare = 0, build = 0, execute = 0, sample = 0, cleanup = 0;
    Clock::time_point ready;
    int32_t token = -1;
};

class Runner {
    ModelBase& model_;
    Backend& backend_;
    bool cuda_;
    std::vector<Backend*> devices_;
    std::vector<int32_t> input_, positions_;
    KVCacheManager cache_;
    GraphContext context_;
    Scheduler scheduler_;
    SequenceId sequence_ = kInvalidSequenceId;
    bool pending_ = false;
    Tensor *tokens_, *pos_;
    Sampler sampler_;
public:
    Runner(ModelBase& model, Backend& backend, Backend* gpu, const std::vector<KVLayerSpec>& layers, size_t capacity)
        : model_(model), backend_(backend), cuda_(gpu != nullptr),
          devices_(gpu ? std::vector<Backend*>{gpu, &backend} : std::vector<Backend*>{&backend}),
          cache_(std::make_unique<ContiguousKVStrategy>(std::make_unique<NaiveKVStorage>())),
          context_(&backend), scheduler_(devices_) {
        std::vector<Backend*> placement(layers.size(), gpu ? gpu : &backend);
        KVCacheConfig config{.layers = layers.size(), .max_tokens_per_sequence = capacity};
        check(cache_.init(layers, placement, config), "初始化 KV 失败");
        tokens_ = context_.make_tensor("bench_tokens", DataType::GGML_TYPE_I32, {1});
        pos_ = context_.make_tensor("bench_positions", DataType::GGML_TYPE_I32, {1});
        for (auto* t : {tokens_, pos_}) {
            t->tensorRole = TensorRole::TNESOR_ROLE_INPUT;
            t->backend = &backend_;
            t->device = DeviceType::CPU;
        }
        context_.set_input_tokens(tokens_);
        context_.set_input_pos(pos_);
    }

    ~Runner() {
        (void)context_.release_kv_access();
        if (pending_) (void)cache_.abort_append(sequence_);
        if (sequence_ != kInvalidSequenceId) (void)cache_.release_sequence(sequence_);
    }

    void begin_request() {
        if (sequence_ != kInvalidSequenceId) throw std::runtime_error("上一请求未释放");
        sequence_ = take(cache_.create_sequence(), "创建 KV 序列失败");
        check(context_.set_kv_cache(&cache_, sequence_), "绑定 KV 失败");
    }

    void end_request() {
        check(context_.release_kv_access(), "归还 KV 失败");
        check(cache_.release_sequence(sequence_), "释放 KV 序列失败");
        sequence_ = kInvalidSequenceId;
        check(context_.set_kv_cache(nullptr, kInvalidSequenceId), "解绑 KV 失败");
    }

    Step evaluate(const std::vector<int32_t>& input, bool sample) {
        Step s;
        const auto start = Clock::now();
        const auto n_past = take(cache_.length(sequence_), "查询 KV 长度失败");
        check(cache_.reserve_append(sequence_, input.size()), "预留 KV 失败");
        pending_ = true;
        input_ = input;
        positions_.resize(input.size());
        std::iota(positions_.begin(), positions_.end(), static_cast<int32_t>(n_past));
        const auto n = static_cast<int64_t>(input.size());
        const int64_t bytes = n * sizeof(int32_t);
        for (auto* t : {tokens_, pos_}) {
            t->dims = {n, 1, 1, 1};
            t->strides = {sizeof(int32_t), bytes, bytes, bytes};
        }
        tokens_->data = input_.data();
        pos_->data = positions_.data();
        const auto prepared = Clock::now();
        s.prepare = seconds(start, prepared);
        {
            auto graph = take(model_.build_graph(context_), "建图失败");
            check(graph.compute(), "拓扑排序失败");
            const auto built = Clock::now();
            s.build = seconds(prepared, built);
            // 包含调度、分配、传输和 GPU 完成等待，不是纯 kernel 时间。
            auto execution = scheduler_.execute(graph);
            const auto sync_error = cuda_ ? cudaDeviceSynchronize() : cudaSuccess;
            check(execution, "调度执行失败");
            if (sync_error != cudaSuccess)
                throw std::runtime_error(std::string("CUDA 执行失败：") + cudaGetErrorString(sync_error));
            const auto executed = Clock::now();
            s.execute = seconds(built, executed);
            if (sample) {
                if (graph.outputs().size() != 1 || !graph.outputs().front())
                    throw std::runtime_error("预期一个 logits 输出");
                auto logits = take(read_logits(*graph.outputs().front(), input.size()-1), "读取 logits 失败");
                s.token = take(sampler_.sample(logits), "采样失败");
            }
            s.ready = Clock::now();
            s.sample = sample ? seconds(executed, s.ready) : 0;
            check(context_.release_kv_access(), "归还 KV 访问失败");
            check(cache_.commit_append(sequence_), "提交 KV 失败");
            pending_ = false;
        }
        s.cleanup = seconds(s.ready, Clock::now());
        return s;
    }
};

struct Result {
    std::string test;
    size_t prompt = 0, generated = 0, evaluated_tokens = 0;
    double wall = 0, prepare = 0, build = 0, execute = 0, sample = 0, cleanup = 0;
    std::optional<double> first, after_first, tpot, request;
    double peak_rss_mib = 0;
    uint64_t checksum = 0;
    double throughput() const { return (test == "pp" ? prompt : generated)/wall; }
};

double peak_rss_mib() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage 失败");
    // Linux ru_maxrss 单位为 KiB，统计整个进程的 RSS 高水位（包括模型加载）。
    return static_cast<double>(usage.ru_maxrss)/1024.0;
}

Result run(Runner& runner, const std::vector<int32_t>& prompt, size_t generated) {
    Result r;
    r.test = generated ? "pg-kv" : "pp";
    r.prompt = prompt.size(); r.generated = generated;
    auto ids = prompt;
    const auto start = Clock::now();
    runner.begin_request();
    Clock::time_point first{}, last{};
    const size_t steps = generated ? generated : 1;
    for (size_t i=0; i<steps; ++i) {
        Step s;
        try {
            s = runner.evaluate(ids, generated != 0);
        } catch (const std::exception& error) {
            throw std::runtime_error("已完成 " + std::to_string(i) + "/" +
                std::to_string(steps) + " 步，当前输入 " + std::to_string(ids.size()) +
                " token：" + error.what());
        }
        r.evaluated_tokens += ids.size();
        r.prepare += s.prepare; r.build += s.build; r.execute += s.execute;
        r.sample += s.sample; r.cleanup += s.cleanup;
        if (generated) {
            if (i == 0) first = s.ready;
            last = s.ready;
            if (i == 0 || (i+1)%16 == 0 || i+1 == generated)
                std::cerr << "进度 " << i+1 << '/' << generated
                          << "，已用 " << seconds(start, s.ready) << " s\n";
            // 固定工作量基准，不是聊天会话：EOS 也作为下一轮输入，保证每轮 N 相同。
            ids.assign(1, s.token);
            r.checksum = r.checksum*131 + static_cast<uint32_t>(s.token);
        }
    }
    runner.end_request();
    r.wall = seconds(start, Clock::now());
    if (generated) r.first = seconds(start, first);
    if (generated) r.request = seconds(start, last);
    if (generated > 1) {
        r.tpot = seconds(first, last)/(generated-1);
        r.after_first = 1.0 / *r.tpot;
    }
    r.peak_rss_mib = peak_rss_mib();
    return r;
}

std::pair<double, double> mean_stdev(const std::vector<double>& values) {
    double mean = std::accumulate(values.begin(), values.end(), 0.0)/values.size();
    double sum = 0;
    for (double v : values) sum += (v-mean)*(v-mean);
    return {mean, values.size()>1 ? std::sqrt(sum/(values.size()-1)) : 0.0};
}

void summary(std::ostream& out, const std::vector<Result>& results) {
    auto show = [&](const char* label, auto metric) {
        std::vector<double> values;
        for (const auto& r : results) values.push_back(metric(r));
        auto [mean, sd] = mean_stdev(values);
        out << "  " << label << ": " << mean;
        if (values.size()>1) out << " ± " << sd;
        out << '\n';
    };
    out << results.front().test << " 汇总（" << results.size() << " 轮，均值 ± 样本标准差）\n";
    show("整轮吞吐量 token/s", [](const Result& r) { return r.throughput(); });
    show("建图/拓扑 ms", [](const Result& r) { return r.build*1000; });
    show("调度执行 ms", [](const Result& r) { return r.execute*1000; });
    show("读取 logits/采样 ms", [](const Result& r) { return r.sample*1000; });
    show("清理 ms", [](const Result& r) { return r.cleanup*1000; });
    if (results.front().test == "pp")
        show("调度执行阶段输入 token/s", [](const Result& r) { return r.prompt/r.execute; });
    if (results.front().first)
        show("TTFT ms（引擎）", [](const Result& r) { return *r.first*1000; });
    if (results.front().tpot)
        show("TPOT / Decode ms/token", [](const Result& r) { return *r.tpot*1000; });
    if (results.front().request)
        show("请求总耗时 s（至最后 token）", [](const Result& r) { return *r.request; });
    out << "  进程峰值 RSS MiB: " << results.back().peak_rss_mib << '\n';
    if (results.front().after_first)
        show("首 token 后生成 token/s（KV cache）", [](const Result& r) { return *r.after_first; });
    if (results.front().generated && std::any_of(results.begin(), results.end(), [&](const Result& r) {
            return r.checksum != results.front().checksum;
        })) out << "  注意：各轮输出 token 校验值不一致。\n";
}

std::string cpu_name() {
    std::ifstream file("/proc/cpuinfo");
    for (std::string line; std::getline(file,line);)
        if (line.starts_with("model name")) return line.substr(line.find(':')+2);
    return "unknown";
}

void csv_row(std::ostream& csv, const Result& r, size_t repetition, size_t threads) {
    csv << r.test << ',' << repetition << ',' << threads << ',' << r.prompt << ',' << r.generated
        << ',' << r.wall << ',' << r.prepare << ',' << r.build << ',' << r.execute << ',' << r.sample
        << ',' << r.cleanup << ',' << r.throughput() << ',';
    if (r.first) csv << *r.first;
    csv << ',';
    if (r.after_first) csv << *r.after_first;
    csv << ',';
    if (r.tpot) csv << *r.tpot*1000;
    csv << ',';
    if (r.request) csv << *r.request;
    csv << ',' << r.peak_rss_mib << ',' << r.checksum << ',' << r.evaluated_tokens << '\n';
    csv.flush();
    if (!csv) throw std::runtime_error("写入 CSV 失败");
}

int benchmark(const Options& o) {
    if (setenv("INFERLLM_THREADS", std::to_string(o.threads).c_str(), 1) != 0)
        throw std::runtime_error("无法设置 CPU 线程数");
    // report 绕过 QuietOutput，只有 benchmark 自身打印表格。
    std::ostream report(std::cout.rdbuf());
    report << std::fixed << std::setprecision(3);
    QuietOutput quiet;
    const auto load_start = Clock::now();
    register_all_backends();
    auto& registry = BackendRegistry::instance();
    registry.probe_all();
    auto backend = take(registry.create("cpu", PoolConfig{}), "创建 CPU 失败");
    std::unique_ptr<Backend> gpu;
    if (o.backend == "cuda:0")
        gpu = take(registry.create("cuda:0", PoolConfig{}), "创建 CUDA 失败");
    const auto* cpu = dynamic_cast<const CPUBackend*>(backend.get());
    if (!cpu) throw std::runtime_error("预期 CPUBackend");
    const size_t actual_threads = static_cast<size_t>(cpu->get_n_threads());
    auto gguf = take(GGUFContext::load(o.model), "加载 GGUF 失败");
    const auto arch = take(gguf.metadata().get_meta<std::string>("general.architecture"), "读取架构失败");
    const auto context = take(gguf.metadata().get_meta<uint32_t>(arch+".context_length"), "读取上下文长度失败");
    const auto vocab = take(gguf.metadata().get_meta_array_len("tokenizer.ggml.tokens"), "读取词表长度失败");
    if (!vocab || vocab > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("词表长度不支持");
    const size_t extra = o.mode == "pp" ? 0 : o.generated-1;
    if (o.prompt > context || extra > context-o.prompt ||
        o.prompt+extra > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("输入长度 + 生成次数 - 1 超过模型上下文长度");
    for (const auto& tensor : gguf.tensors())
        if (tensor._type != DataType::GGML_TYPE_F32)
            throw std::runtime_error("当前 benchmark 仅支持 F32 模型，请提供 F32 GGUF");
    const auto layer_count = take(gguf.metadata().get_meta<uint32_t>(arch+".block_count"), "读取层数失败");
    const auto heads = take(gguf.metadata().get_meta<uint32_t>(arch+".attention.head_count"), "读取头数失败");
    const auto kv_heads = take(gguf.metadata().get_meta<uint32_t>(arch+".attention.head_count_kv"), "读取 KV 头数失败");
    const auto embd = take(gguf.metadata().get_meta<uint32_t>(arch+".embedding_length"), "读取维度失败");
    if (!layer_count || !heads || !kv_heads || !embd) throw std::runtime_error("模型 KV 形状无效");
    const auto kd = gguf.metadata().get_meta<uint32_t>(arch+".attention.key_length").value_or(embd/heads);
    const auto vd = gguf.metadata().get_meta<uint32_t>(arch+".attention.value_length").value_or(embd/heads);
    if (!kd || !vd) throw std::runtime_error("KV head 维度无效");
    std::vector<KVLayerSpec> layers(layer_count, KVLayerSpec{
        .n_kv_heads=kv_heads, .k_head_dim=kd, .v_head_dim=vd,
        .k_type=GGML_TYPE_F32, .v_type=GGML_TYPE_F32});
    register_all_models();
    auto model = take(ModelBase::load_model(std::move(gguf)), "创建模型失败");
    check(model->load_hparams(), "读取超参数失败");
    check(model->load_weight(), "加载权重失败");
    std::vector<Backend*> devices;
    if (gpu) devices.push_back(gpu.get());
    devices.push_back(backend.get());
    PlacementConfig placement;
    placement.n_ctx = static_cast<uint32_t>(o.prompt+extra);
    check(model->plan_weights_placement(devices, placement), "权重放置失败");
    const double load_seconds = seconds(load_start, Clock::now());
    std::vector<int32_t> prompt(o.prompt);
    for (size_t i=0; i<prompt.size(); ++i) prompt[i] = static_cast<int32_t>((100+i)%vocab);
    Runner runner(*model, *backend, gpu.get(), layers, o.prompt+extra);
    std::ofstream csv;
    if (!o.csv.empty()) {
        const std::filesystem::path path(o.csv);
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        csv.open(path);
        if (!csv) throw std::runtime_error("无法打开 CSV 文件："+o.csv);
        csv << std::setprecision(12)
            << "test,run,threads,prompt_tokens,generated_tokens,wall_s,prepare_s,build_s,execute_s,sample_s,cleanup_s,tokens_per_second,ttft_s,decode_tokens_per_second,tpot_ms,request_s,peak_rss_mib,token_checksum,evaluated_tokens\n";
    }
    report << "Backend: " << o.backend << " (CPU embedding; CUDA synchronized when enabled)\n";
    report << "CPU: " << cpu_name() << "\nModel: " << o.model
           << "\nEngine git: " << BENCH_GIT_COMMIT
           << "\nProfile: " << (o.profile.empty() ? "custom" : o.profile)
           << "\nBuild: " << BENCH_BUILD_TYPE << "; compiler: " << __VERSION__
           << "\nThreads: " << actual_threads << (o.threads == 0 ? " (auto)" : " (explicit)")
           << "; input: " << o.prompt << "; generated: " << o.generated
           << "; warmup: " << o.warmup << "; repeats: " << o.repeats
           << "\nSetup: " << load_seconds << " s（不计入测试）\n"
           << "固定合成输入，完成同步后计时；pg 使用 KV Cache；首轮处理 P 个输入，随后每轮只处理 1 个 token。\n";
    if (std::string_view(BENCH_BUILD_TYPE) != "Release") report << "注意：当前不是 Release 构建。\n";
    for (const auto& mode : {std::string("pp"), std::string("pg")}) {
        if (o.mode != "all" && o.mode != mode) continue;
        const size_t n = mode == "pp" ? 0 : o.generated;
        for (size_t i=0; i<o.warmup; ++i) {
            report << "预热 " << mode << ' ' << i+1 << '/' << o.warmup << std::endl;
            (void)run(runner, prompt, n);
        }
        std::vector<Result> results;
        for (size_t i=0; i<o.repeats; ++i) {
            report << "测试 " << mode << ' ' << i+1 << '/' << o.repeats << std::endl;
            auto r = run(runner, prompt, n);
            report << "  wall=" << r.wall*1000 << " ms; build=" << r.build*1000
                   << " ms; execute=" << r.execute*1000 << " ms; sample=" << r.sample*1000
                   << " ms; throughput=" << r.throughput() << " token/s\n";
            if (csv.is_open()) csv_row(csv, r, i+1, actual_threads);
            results.push_back(std::move(r));
        }
        summary(report, results);
    }
    if (csv.is_open()) report << "CSV: " << o.csv << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.help) { help(); return 0; }
        return benchmark(options);
    } catch (const std::exception& error) {
        std::cerr << "benchmark 失败：" << error.what() << '\n';
        std::cerr << "失败时进程峰值 RSS MiB: " << peak_rss_mib() << '\n';
        return 1;
    }
}
