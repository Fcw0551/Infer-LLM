#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

class GGUFMetadata;
namespace tokenizer_detail { struct Data; }

enum class TokenizerError {
    MissingMetadata = 1, InvalidMetadata, UnsupportedStrategy,
    InvalidUtf8, InvalidToken, InvalidOptions, RegexFailure
};
std::error_code make_error_code(TokenizerError error) noexcept;

struct EncodeOptions {
    // CONTROL/UNKNOWN 仅在开启时识别；USER_DEFINED 始终作为原子 token。
    bool parse_special = false;
    // nullopt 表示采用 GGUF 配置；显式 true/false 覆盖配置。
    std::optional<bool> add_bos;
    std::optional<bool> add_eos;
};

struct DecodeOptions {
    bool skip_special = true; // 隐藏 CONTROL/UNKNOWN，不隐藏 USER_DEFINED。
};

// 独立持有不可变词表；不借用 GGUF 内存，不依赖模型、图或后端。
// 支持并发只读调用。返回的 string 是原始字节，可包含 NUL/不完整 UTF-8。
class Tokenizer {
public:
    static std::expected<Tokenizer, std::error_code>
    from_gguf(const GGUFMetadata& metadata);

    // Qwen2/Qwen3 普通文本先做 NFC 规范化；无效 UTF-8 返回错误。
    std::expected<std::vector<int32_t>, std::error_code>
    encode(std::string_view text, EncodeOptions options = {}) const;

    std::expected<std::string, std::error_code>
    decode(std::span<const int32_t> tokens, DecodeOptions options = {}) const;

    // 返回原始片段，包括特殊 token 的字面内容；拼接后才能解码 UTF-8。
    std::expected<std::string, std::error_code> token_to_piece(int32_t token) const;
    size_t vocab_size() const noexcept;
    bool is_eog(int32_t token) const noexcept;
    std::optional<int32_t> bos_id() const noexcept;
    std::optional<int32_t> eos_id() const noexcept;

private:
    explicit Tokenizer(std::shared_ptr<const tokenizer_detail::Data> data)
        : data_(std::move(data)) {}
    std::shared_ptr<const tokenizer_detail::Data> data_;
};
