#include "sampler.hpp"


// ------- 读取logits，后续给sampler ---------------
std::expected<std::vector<float>, std::error_code> read_logits(const Tensor& tensor, size_t token_index) {
    if (tensor.dataType != DataType::GGML_TYPE_F32){ //todo
        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }
    if (!tensor.data || !tensor.backend || tensor.dims[0] <= 0 || tensor.dims[1] <= 0 || tensor.dims[2] != 1 || tensor.dims[3] != 1){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (token_index >= static_cast<size_t>(tensor.dims[1])){
        return std::unexpected(std::make_error_code(std::errc::result_out_of_range));
    }
    const size_t vocab = static_cast<size_t>(tensor.dims[0]);
    const size_t count = static_cast<size_t>(tensor.dims[1]);
    const size_t max_bytes = static_cast<size_t>(std::numeric_limits<int64_t>::max());  //int64_t 能存的最大值
    if (vocab > max_bytes / sizeof(float) || count > max_bytes / (vocab * sizeof(float))){
        return std::unexpected(std::make_error_code(std::errc::value_too_large));
    }
    const size_t row_bytes = vocab * sizeof(float);
    const size_t total_bytes = row_bytes * count;
    if (tensor.strides[0] != sizeof(float) ||
        tensor.strides[1] != static_cast<int64_t>(row_bytes) ||
        tensor.strides[2] != static_cast<int64_t>(total_bytes) ||
        tensor.strides[3] != static_cast<int64_t>(total_bytes)){
        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }

    std::vector<float> logits(vocab);
    auto result = tensor.backend->copy_tensor_get(&tensor, logits.data(), token_index * row_bytes, row_bytes);
    if (!result){
        return std::unexpected(result.error());
    }
    return logits;
}


// ----------------- sampler -------------------
namespace {

std::expected<int32_t, std::error_code> sample_greedy(std::span<const float> logits) {
    if (logits.empty()) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (logits.size() - 1 > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return std::unexpected(std::make_error_code(std::errc::value_too_large));
    }
    float best_score = -std::numeric_limits<float>::infinity();
    int32_t best_id = -1;
    for (size_t i = 0; i < logits.size(); ++i) {
        const float score = logits[i];
        if (std::isnan(score) || score == std::numeric_limits<float>::infinity()) {
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        }
        if (score > best_score) {
            best_score = score;
            best_id = static_cast<int32_t>(i);
        }
    }
    if (best_id < 0) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    return best_id;
}

} // namespace

std::expected<int32_t, std::error_code> Sampler::sample(std::span<const float> logits) const {
    switch (_config.mode) {
    case SamplingMode::Greedy:
        return sample_greedy(logits);
    }

    // 不设置 default，让编译器提示新增但尚未处理的枚举策略。
    return std::unexpected(std::make_error_code(std::errc::not_supported));
}
