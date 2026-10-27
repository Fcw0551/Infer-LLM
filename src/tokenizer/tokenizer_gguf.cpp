#include "tokenizer_data.hpp"
#include "gguf_parser.hpp"
#include <limits>

namespace tokenizer_detail {
namespace {
auto invalid() { return std::unexpected(make_error_code(TokenizerError::InvalidMetadata)); }

template<class T>
std::expected<T, std::error_code> scalar(const GGUFMetadata& m, const char* key, GGUFType type) {
    auto v = m.find(key);
    if (!v) return std::unexpected(make_error_code(TokenizerError::MissingMetadata));
    if (v->type != type) return invalid();
    auto p = std::get_if<T>(&v->scalar);
    if (!p) return invalid();
    return *p;
}
std::expected<const MetaValue*, std::error_code>
array(const GGUFMetadata& m, const char* key, GGUFType type) {
    auto v = m.find(key);
    if (!v) return std::unexpected(make_error_code(TokenizerError::MissingMetadata));
    if (v->type != GGUF_TYPE_ARRAY || v->array_elem_type != type) return invalid();
    return v;
}
std::string utf8(unsigned cp) {
    if (cp < 128) return std::string(1, static_cast<char>(cp));
    return {static_cast<char>(0xc0 | (cp >> 6)), static_cast<char>(0x80 | (cp & 63))};
}
}

std::expected<std::shared_ptr<const Data>, std::error_code> load_gguf(const GGUFMetadata& m) {
    auto model = scalar<std::string>(m, "tokenizer.ggml.model", GGUF_TYPE_STRING);
    auto pre = scalar<std::string>(m, "tokenizer.ggml.pre", GGUF_TYPE_STRING);
    if (!model) return std::unexpected(model.error());
    if (!pre) return std::unexpected(pre.error());
    if (*model != "gpt2" || *pre != "qwen2")
        return std::unexpected(make_error_code(TokenizerError::UnsupportedStrategy));

    auto tokens = array(m, "tokenizer.ggml.tokens", GGUF_TYPE_STRING);
    auto types = array(m, "tokenizer.ggml.token_type", GGUF_TYPE_INT32);
    auto merges = array(m, "tokenizer.ggml.merges", GGUF_TYPE_STRING);
    if (!tokens) return std::unexpected(tokens.error());
    if (!types) return std::unexpected(types.error());
    if (!merges) return std::unexpected(merges.error());
    const size_t n = (*tokens)->array_elems.size();
    if (n == 0 || n-1 > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        (*types)->array_elems.size() != n) return invalid();
    auto d = std::make_shared<Data>();
    d->tokens.reserve(n); d->types.reserve(n); d->pieces.resize(n); d->ids.reserve(n);
    for (size_t i=0; i<n; ++i) {
        auto token = std::get_if<std::string>(&(*tokens)->array_elems[i]);
        auto type = std::get_if<int32_t>(&(*types)->array_elems[i]);
        if (!token || token->empty() || !type || *type < 1 || *type > 6) return invalid();
        // Byte token spelling belongs to other tokenizer families, not this byte-BPE profile.
        if (*type == static_cast<int32_t>(TokenType::Byte))
            return std::unexpected(make_error_code(TokenizerError::UnsupportedStrategy));
        if (!d->ids.emplace(*token, static_cast<int32_t>(i)).second) return invalid();
        d->tokens.push_back(*token);
        d->types.push_back(static_cast<TokenType>(*type));
    }

    // GPT-2 可逆字节字母表：可打印 Latin-1 保留，其余映射到 U+0100 起。
    std::unordered_map<std::string, unsigned char> inverse;
    unsigned extra = 256;
    for (unsigned b=0; b<256; ++b) {
        const bool direct = (b>=33 && b<=126) || (b>=161 && b<=172) || b>=174;
        auto symbol = utf8(direct ? b : extra++);
        inverse.emplace(symbol, static_cast<unsigned char>(b));
        auto it = d->ids.find(symbol);
        if (it == d->ids.end() || d->types[it->second] != TokenType::Normal) return invalid();
        d->byte_ids[b] = it->second;
    }
    for (size_t i=0; i<n; ++i) {
        auto type = d->types[i];
        const auto& token = d->tokens[i];
        if (type == TokenType::Unused) continue;
        if (type == TokenType::Normal) {
            for (size_t p=0; p<token.size();) {
                size_t len = static_cast<unsigned char>(token[p]) < 128 ? 1 : 2;
                auto it = inverse.find(token.substr(p, len));
                if (it == inverse.end()) return invalid();
                d->pieces[i].push_back(static_cast<char>(it->second));
                p += len;
            }
        } else {
            d->pieces[i] = token;
            // 校验新增 token 的 UTF-8，保证匹配不会跳过无效编码。
            if (auto r = pre_tokenize(token, d->pre); !r) return invalid();
            size_t node = 0;
            for (unsigned char ch : token) {
                auto it = d->added[node].next.find(ch);
                if (it != d->added[node].next.end()) { node = it->second; continue; }
                const size_t next = d->added.size();
                d->added[node].next.emplace(ch, next);
                d->added.emplace_back();
                node = next;
            }
            d->added[node].id = static_cast<int32_t>(i);
        }
    }
    for (size_t rank=0; rank<(*merges)->array_elems.size(); ++rank) {
        auto entry = std::get_if<std::string>(&(*merges)->array_elems[rank]);
        if (!entry) return invalid();
        auto pos = entry->find(' ');
        if (pos == std::string::npos || pos == 0 || pos+1 == entry->size() ||
            entry->find(' ',pos+1) != std::string::npos) return invalid();
        auto left = entry->substr(0,pos), right = entry->substr(pos+1);
        auto l = d->ids.find(left), r = d->ids.find(right), out = d->ids.find(left+right);
        if (l == d->ids.end() || r == d->ids.end() || out == d->ids.end()) return invalid();
        if (d->types[l->second] != TokenType::Normal || d->types[r->second] != TokenType::Normal ||
            d->types[out->second] != TokenType::Normal) return invalid();
        if (!d->merges.emplace(pair_key(l->second,r->second), Merge{rank,out->second}).second) return invalid();
    }
    auto id = [&](const char* key) -> std::expected<std::optional<int32_t>, std::error_code> {
        if (!m.find(key)) return std::nullopt;
        auto value = scalar<uint32_t>(m,key,GGUF_TYPE_UINT32);
        if (!value) return std::unexpected(value.error());
        if (*value >= n || d->types[*value] == TokenType::Unused) return invalid();
        return static_cast<int32_t>(*value);
    };
    auto bos = id("tokenizer.ggml.bos_token_id"), eos = id("tokenizer.ggml.eos_token_id");
    if (!bos) return std::unexpected(bos.error());
    if (!eos) return std::unexpected(eos.error());
    d->bos = *bos; d->eos = *eos;
    for (auto key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
        auto value = id(key);
        if (!value) return std::unexpected(value.error());
        if (*value) d->eog.insert(**value);
    }
    // Qwen2 profile 的文档结束标记也是 EOG；从词表查 ID，不硬编码编号。
    if (auto it=d->ids.find("<|endoftext|>"); it!=d->ids.end() && is_special(d->types[it->second]))
        d->eog.insert(it->second);
    for (auto [key, dest] : {std::pair{"tokenizer.ggml.add_bos_token", &d->add_bos},
                            std::pair{"tokenizer.ggml.add_eos_token", &d->add_eos}}) {
        if (!m.find(key)) continue;
        auto flag = scalar<bool>(m,key,GGUF_TYPE_BOOL);
        if (!flag) return std::unexpected(flag.error());
        *dest = *flag;
    }
    if ((d->add_bos && !d->bos) || (d->add_eos && !d->eos)) return invalid();
    return std::shared_ptr<const Data>(std::move(d));
}
} // namespace tokenizer_detail
