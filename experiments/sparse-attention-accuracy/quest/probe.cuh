// Included inside namespace CUDA_F32 in the experimental kernel build.
namespace quest_probe {
std::FILE* trace = nullptr;
const char* phase = nullptr;
int block_size = 0;
int budget = 0;
int layer = 0;

__global__ void bounds(const float* q, const float* k, float* result,
                       int length, int heads, int kv_heads, int dim, int size) {
    const int pages = (length + size - 1) / size;
    const int head = blockIdx.x / pages;
    const int page = blockIdx.x % pages;
    const int kv_head = head / (heads / kv_heads);
    const int begin = page * size;
    const int end = min(begin + size, length);
    float local = 0;
    for (int d = threadIdx.x; d < dim; d += 32) {
        float low = CUDART_INF_F, high = -CUDART_INF_F;
        for (int t = begin; t < end; ++t) {
            const float key = k[(t * kv_heads + kv_head) * dim + d];
            low = fminf(low, key);
            high = fmaxf(high, key);
        }
        const float query = q[head * dim + d];
        local += fmaxf(query * low, query * high);
    }
    for (int delta = 16; delta; delta /= 2)
        local += __shfl_down_sync(0xffffffff, local, delta);
    if (threadIdx.x == 0) result[blockIdx.x] = local;
}

__global__ void mask(float* scores, const uint8_t* selected, int length, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (i < length && !selected[head * ((length + size - 1) / size) + i / size])
        scores[head * length + i] = -CUDART_INF_F;
}

std::vector<int> order(const float* scores, int count, int mandatory = -1) {
    std::vector<int> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        if (a == mandatory || b == mandatory) return a == mandatory && a != b;
        return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
    });
    return indices;
}

cudaError_t ranks(const float* q, const float* k, int length, int heads,
                  int kv_heads, int dim, int size, cudaStream_t stream,
                  std::vector<float>& result) {
    const int count = heads * ((length + size - 1) / size);
    result.resize(count);
    float* device = nullptr;
    auto status = cudaMalloc(&device, count * sizeof(float));
    if (status != cudaSuccess) return status;
    bounds<<<count, 32, 0, stream>>>(q, k, device, length, heads, kv_heads, dim, size);
    status = cudaGetLastError();
    if (status == cudaSuccess)
        status = cudaMemcpyAsync(result.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    const auto sync = cudaStreamSynchronize(stream);
    const auto freed = cudaFree(device);
    if (status != cudaSuccess) return status;
    return sync == cudaSuccess ? freed : sync;
}

cudaError_t select(const float* q, const float* k, float* scores, int length,
                   int heads, int kv_heads, int dim, cudaStream_t stream) {
    const int size = block_size;
    const int pages = (length + size - 1) / size;
    const int limit = std::min(pages, budget / size);
    if (limit == pages) return cudaSuccess;
    std::vector<float> ranking;
    auto status = ranks(q, k, length, heads, kv_heads, dim, size, stream, ranking);
    if (status != cudaSuccess) return status;
    std::vector<uint8_t> selected(heads * pages, 0);
    for (int h = 0; h < heads; ++h) {
        auto indices = order(ranking.data() + h * pages, pages, pages - 1);
        for (int i = 0; i < limit; ++i) selected[h * pages + indices[i]] = 1;
    }
    uint8_t* device = nullptr;
    status = cudaMalloc(&device, selected.size());
    if (status != cudaSuccess) return status;
    status = cudaMemcpyAsync(device, selected.data(), selected.size(), cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) {
        mask<<<dim3((length + 127) / 128, heads), 128, 0, stream>>>(scores, device, length, size);
        status = cudaGetLastError();
    }
    const auto sync = cudaStreamSynchronize(stream);
    const auto freed = cudaFree(device);
    if (status != cudaSuccess) return status;
    return sync == cudaSuccess ? freed : sync;
}

cudaError_t capture(const float* q, const float* k, const float* scores, int length,
                    int heads, int kv_heads, int dim, cudaStream_t stream) {
    if (!trace || !phase) return cudaSuccess;
    const int current_layer = layer++;
    // Four layer depths, and one in eight query positions.
    if (length % 8 || (current_layer != 0 && current_layer != 9 &&
                      current_layer != 18 && current_layer != 27)) return cudaSuccess;
    std::vector<float> weights(length * heads);
    auto status = cudaMemcpyAsync(weights.data(), scores, weights.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) return status;
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) return status;
    constexpr int budgets[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    auto emit = [&](int head, const char* method, int size, int requested, int read, float mass) {
        std::fprintf(trace, "%s,%d,%d,%d,%s,%d,%d,%d,%.8f\n",
                     phase, length, current_layer, head, method, size, requested, read, mass);
    };
    for (int h = 0; h < heads; ++h) {
        const float* row = weights.data() + h * length;
        auto indices = order(row, length);
        for (int b : budgets) {
            const int read = std::min(length, b);
            float mass = 0;
            for (int i = 0; i < read; ++i) mass += row[indices[i]];
            emit(h, "token", 1, b, read, mass);
        }
    }
    for (int size : {16, 32, 64}) {
        const int pages = (length + size - 1) / size;
        std::vector<float> ranking;
        status = ranks(q, k, length, heads, kv_heads, dim, size, stream, ranking);
        if (status != cudaSuccess) return status;
        // Future offload can share loads within each GQA group.
        const int group = heads / kv_heads;
        for (int kv = 0; kv < kv_heads; ++kv) {
            std::vector<std::vector<int>> ranked;
            for (int h = kv * group; h < (kv + 1) * group; ++h)
                ranked.push_back(order(ranking.data() + h * pages, pages, pages - 1));
            for (int b : budgets) {
                if (b < size) continue;
                std::vector<bool> selected(pages, false);
                for (const auto& indices : ranked)
                    for (int i = 0; i < std::min(pages, b / size); ++i)
                        selected[indices[i]] = true;
                int read = 0;
                for (int p = 0; p < pages; ++p)
                    if (selected[p]) read += std::min(size, length - p * size);
                emit(kv, "union", size, b, read, 0);
            }
        }
        for (int h = 0; h < heads; ++h) {
            std::vector<float> mass(pages, 0);
            for (int i = 0; i < length; ++i) mass[i / size] += weights[h * length + i];
            for (bool quest : {false, true}) {
                auto indices = order(quest ? ranking.data() + h * pages : mass.data(), pages, pages - 1);
                for (int b : budgets) {
                    if (b < size) continue;
                    float kept = 0;
                    int read = 0;
                    for (int i = 0; i < std::min(pages, b / size); ++i) {
                        const int page = indices[i];
                        kept += mass[page];
                        read += std::min(size, length - page * size);
                    }
                    emit(h, quest ? "quest" : "oracle", size, b, read, kept);
                }
            }
        }
    }
    return std::ferror(trace) ? cudaErrorUnknown : cudaSuccess;
}
}

void quest_experiment_configure(std::FILE* trace, const char* phase, int size, int budget) {
    quest_probe::trace = trace;
    quest_probe::phase = phase;
    quest_probe::block_size = size;
    quest_probe::budget = budget;
    quest_probe::layer = 0;
}
