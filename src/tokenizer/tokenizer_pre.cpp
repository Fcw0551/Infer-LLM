#include "tokenizer_data.hpp"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <limits>

namespace tokenizer_detail {
namespace {
struct Regex {
    pcre2_code* code = nullptr;
    Regex() {
        // Qwen2/Qwen3 tokenizer.json 的预分词规则。显式 ASCII 大小写收缩词。
        constexpr auto pattern = R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
        int error;
        PCRE2_SIZE offset;
        code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern), PCRE2_ZERO_TERMINATED,
                             PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr);
    }
    ~Regex() { pcre2_code_free(code); }
};
}
std::expected<std::string, std::error_code> normalize_text(std::string_view text, PreType type) {
    switch (type) {
    case PreType::Qwen2: break;
    default: return std::unexpected(make_error_code(TokenizerError::UnsupportedStrategy));
    }
    if (text.empty()) return std::string{};
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        return std::unexpected(std::make_error_code(std::errc::value_too_large));
    UErrorCode status = U_ZERO_ERROR;
    int32_t length = 0;
    u_strFromUTF8(nullptr, 0, &length, text.data(), static_cast<int32_t>(text.size()), &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
        return std::unexpected(make_error_code(TokenizerError::InvalidUtf8));
    std::vector<UChar> input(length);
    status = U_ZERO_ERROR;
    u_strFromUTF8(input.data(), length, nullptr, text.data(), static_cast<int32_t>(text.size()), &status);
    if (U_FAILURE(status)) return std::unexpected(make_error_code(TokenizerError::InvalidUtf8));
    status = U_ZERO_ERROR;
    const auto* nfc = unorm2_getNFCInstance(&status);
    if (U_FAILURE(status)) return std::unexpected(make_error_code(TokenizerError::RegexFailure));
    int32_t needed = unorm2_normalize(nfc, input.data(), length, nullptr, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
        return std::unexpected(make_error_code(TokenizerError::RegexFailure));
    std::vector<UChar> normalized(needed);
    status = U_ZERO_ERROR;
    unorm2_normalize(nfc, input.data(), length, normalized.data(), needed, &status);
    if (U_FAILURE(status)) return std::unexpected(make_error_code(TokenizerError::RegexFailure));
    int32_t bytes = 0;
    status = U_ZERO_ERROR;
    u_strToUTF8(nullptr, 0, &bytes, normalized.data(), needed, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
        return std::unexpected(make_error_code(TokenizerError::RegexFailure));
    std::string output(bytes, '\0');
    status = U_ZERO_ERROR;
    u_strToUTF8(output.data(), bytes, nullptr, normalized.data(), needed, &status);
    if (U_FAILURE(status)) return std::unexpected(make_error_code(TokenizerError::RegexFailure));
    return output;
}
std::expected<std::vector<std::string_view>, std::error_code>
pre_tokenize(std::string_view text, PreType type) {
    switch (type) {
    case PreType::Qwen2: break;
    default: return std::unexpected(make_error_code(TokenizerError::UnsupportedStrategy));
    }
    static const Regex regex;
    if (!regex.code) return std::unexpected(make_error_code(TokenizerError::RegexFailure));
    std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(
        pcre2_match_data_create_from_pattern(regex.code, nullptr), pcre2_match_data_free);
    if (!match) return std::unexpected(std::make_error_code(std::errc::not_enough_memory));
    std::vector<std::string_view> pieces;
    size_t pos = 0;
    while (pos < text.size()) {
        const auto flags = PCRE2_ANCHORED | (pos == 0 ? 0u : PCRE2_NO_UTF_CHECK);
        int rc = pcre2_match(regex.code, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(),
                             pos, flags, match.get(), nullptr);
        if (rc < 0) {
            if (rc <= PCRE2_ERROR_UTF8_ERR1 && rc >= PCRE2_ERROR_UTF8_ERR21)
                return std::unexpected(make_error_code(TokenizerError::InvalidUtf8));
            return std::unexpected(make_error_code(TokenizerError::RegexFailure));
        }
        auto* offsets = pcre2_get_ovector_pointer(match.get());
        if (offsets[0] != pos || offsets[1] <= pos)
            return std::unexpected(make_error_code(TokenizerError::RegexFailure));
        pieces.push_back(text.substr(pos, offsets[1]-pos));
        pos = offsets[1];
    }
    return pieces;
}
} // namespace tokenizer_detail
