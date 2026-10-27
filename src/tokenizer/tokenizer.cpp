#include "tokenizer_data.hpp"

namespace {
class TokenizerCategory final : public std::error_category {
public:
    const char* name() const noexcept override { return "tokenizer"; }
    std::string message(int value) const override {
        switch (static_cast<TokenizerError>(value)) {
        case TokenizerError::MissingMetadata: return "required tokenizer metadata is missing";
        case TokenizerError::InvalidMetadata: return "invalid tokenizer metadata or vocabulary";
        case TokenizerError::UnsupportedStrategy: return "unsupported tokenizer algorithm or pre-tokenizer";
        case TokenizerError::InvalidUtf8: return "input is not valid UTF-8";
        case TokenizerError::InvalidToken: return "invalid or unused token ID";
        case TokenizerError::InvalidOptions: return "requested BOS/EOS is not configured";
        case TokenizerError::RegexFailure: return "Unicode pre-tokenization failed";
        }
        return "unknown tokenizer error";
    }
};
}

std::error_code make_error_code(TokenizerError error) noexcept {
    static TokenizerCategory category;
    return {static_cast<int>(error), category};
}

std::expected<Tokenizer, std::error_code> Tokenizer::from_gguf(const GGUFMetadata& metadata) {
    auto data = tokenizer_detail::load_gguf(metadata);
    if (!data) return std::unexpected(data.error());
    return Tokenizer(std::move(*data));
}

std::expected<std::vector<int32_t>, std::error_code>
Tokenizer::encode(std::string_view text, EncodeOptions options) const {
    using namespace tokenizer_detail;
    if (!data_) return std::unexpected(make_error_code(TokenizerError::InvalidOptions));
    const auto& d = *data_;
    const bool bos = options.add_bos.value_or(d.add_bos);
    const bool eos = options.add_eos.value_or(d.add_eos);
    if ((bos && !d.bos) || (eos && !d.eos))
        return std::unexpected(make_error_code(TokenizerError::InvalidOptions));
    std::vector<int32_t> result;
    if (bos) result.push_back(*d.bos);
    auto ordinary = [&](std::string_view part) -> std::expected<void, std::error_code> {
        switch (d.algorithm) {
        case Algorithm::ByteBPE: return encode_bpe(d, part, result);
        }
        return std::unexpected(make_error_code(TokenizerError::UnsupportedStrategy));
    };
    // 左到右匹配，当前位置优先最长的、当前选项允许的新增 token。
    size_t start = 0, pos = 0;
    while (pos < text.size()) {
        size_t node = 0, end = pos;
        int32_t id = -1;
        for (size_t j = pos; j < text.size(); ++j) {
            auto it = d.added[node].next.find(static_cast<unsigned char>(text[j]));
            if (it == d.added[node].next.end()) break;
            node = it->second;
            const auto candidate = d.added[node].id;
            if (candidate >= 0 && (options.parse_special || !is_special(d.types[candidate]))) {
                id = candidate;
                end = j + 1;
            }
        }
        if (id < 0) { ++pos; continue; }
        if (auto r = ordinary(text.substr(start, pos-start)); !r) return std::unexpected(r.error());
        result.push_back(id);
        start = pos = end;
    }
    if (auto r = ordinary(text.substr(start)); !r) return std::unexpected(r.error());
    if (eos) result.push_back(*d.eos);
    return result;
}

std::expected<std::string, std::error_code> Tokenizer::token_to_piece(int32_t id) const {
    if (!data_ || id < 0 || static_cast<size_t>(id) >= data_->tokens.size() ||
        data_->types[id] == tokenizer_detail::TokenType::Unused)
        return std::unexpected(make_error_code(TokenizerError::InvalidToken));
    return data_->pieces[id];
}

std::expected<std::string, std::error_code>
Tokenizer::decode(std::span<const int32_t> tokens, DecodeOptions options) const {
    std::string result;
    for (int32_t id : tokens) {
        auto piece = token_to_piece(id);
        if (!piece) return std::unexpected(piece.error());
        if (options.skip_special && tokenizer_detail::is_special(data_->types[id])) continue;
        result += *piece;
    }
    return result;
}
size_t Tokenizer::vocab_size() const noexcept { return data_ ? data_->tokens.size() : 0; }
bool Tokenizer::is_eog(int32_t id) const noexcept { return data_ && data_->eog.contains(id); }
std::optional<int32_t> Tokenizer::bos_id() const noexcept { return data_ ? data_->bos : std::nullopt; }
std::optional<int32_t> Tokenizer::eos_id() const noexcept { return data_ ? data_->eos : std::nullopt; }
