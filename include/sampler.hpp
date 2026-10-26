#pragma once
#include <cstdint>
#include <expected>
#include <span>
#include <system_error>
#include <vector>
#include <cmath>
#include <limits>
#include "tensor.hpp"
#include "backend.hpp"
// 读取连续 F32 [n_vocab, n_tokens, 1, 1] 中指定位置的 logits
// 返回的 CPU 数据独立拥有内存，不借用图的激活缓冲区 todo
std::expected<std::vector<float>, std::error_code> read_logits(const Tensor &tensor, size_t token_index);

// 采用的
enum class SamplingMode{
    Greedy              // 贪心（挑选最大的）
};

struct SamplingConfig {
    SamplingMode mode = SamplingMode::Greedy;
};

// 只依赖 CPU logits，不持有模型、计算图或后端。
class Sampler {
public:
    explicit Sampler(SamplingConfig config = {}) : _config(config) {}

    // 同分选择较小 ID。允许 -inf 屏蔽候选，拒绝 NaN/+inf/全部屏蔽。
    std::expected<int32_t, std::error_code> sample(std::span<const float> logits) const;

private:
    SamplingConfig _config;
};
