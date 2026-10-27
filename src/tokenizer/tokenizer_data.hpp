#pragma once
#include "tokenizer.hpp"
#include <array>
#include <unordered_map>
#include <unordered_set>

namespace tokenizer_detail {
enum class Algorithm { ByteBPE };
enum class PreType { Qwen2 };
enum class TokenType : int32_t { Normal=1, Unknown=2, Control=3, UserDefined=4, Unused=5, Byte=6 };
struct TrieNode {
    std::unordered_map<unsigned char, size_t> next;
    int32_t id = -1;
};
struct Merge { size_t rank; int32_t result; };
struct Data {
    Algorithm algorithm = Algorithm::ByteBPE;
    PreType pre = PreType::Qwen2;
    std::vector<std::string> tokens, pieces;
    std::vector<TokenType> types;
    std::unordered_map<std::string, int32_t> ids;
    std::array<int32_t, 256> byte_ids{};
    std::unordered_map<uint64_t, Merge> merges;
    std::vector<TrieNode> added{1};
    std::optional<int32_t> bos, eos;
    std::unordered_set<int32_t> eog;
    bool add_bos = false, add_eos = false;
};
inline uint64_t pair_key(int32_t left, int32_t right) {
    return (uint64_t(uint32_t(left)) << 32) | uint32_t(right);
}
inline bool is_special(TokenType type) {
    return type == TokenType::Control || type == TokenType::Unknown;
}
std::expected<std::shared_ptr<const Data>, std::error_code> load_gguf(const GGUFMetadata&);
std::expected<std::vector<std::string_view>, std::error_code> pre_tokenize(std::string_view, PreType);
std::expected<std::string, std::error_code> normalize_text(std::string_view, PreType);
std::expected<void, std::error_code> encode_bpe(const Data&, std::string_view, std::vector<int32_t>&);
} // namespace tokenizer_detail
