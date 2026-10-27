#include "tokenizer_data.hpp"
#include <queue>

namespace tokenizer_detail {
std::expected<void, std::error_code>
encode_bpe(const Data& data, std::string_view text, std::vector<int32_t>& output) {
    auto normalized = normalize_text(text, data.pre);
    if (!normalized) return std::unexpected(normalized.error());
    auto pieces = pre_tokenize(*normalized, data.pre);
    if (!pieces) return std::unexpected(pieces.error());
    constexpr size_t none = static_cast<size_t>(-1);
    struct Node { int32_t id; size_t prev, next, version = 0; bool live = true; };
    struct Candidate {
        size_t rank, left, right, lv, rv;
        int32_t result;
        bool operator<(const Candidate& other) const {
            return rank > other.rank || (rank == other.rank && left > other.left);
        }
    };
    for (auto piece : *pieces) {
        std::vector<Node> nodes;
        nodes.reserve(piece.size());
        for (size_t i=0; i<piece.size(); ++i)
            nodes.push_back({data.byte_ids[static_cast<unsigned char>(piece[i])],
                             i == 0 ? none : i-1, i+1 == piece.size() ? none : i+1});
        std::priority_queue<Candidate> queue;
        auto push = [&](size_t l) {
            if (l == none || !nodes[l].live || nodes[l].next == none) return;
            const size_t r = nodes[l].next;
            auto it = data.merges.find(pair_key(nodes[l].id, nodes[r].id));
            if (it != data.merges.end())
                queue.push({it->second.rank, l, r, nodes[l].version, nodes[r].version, it->second.result});
        };
        for (size_t i=0; i<nodes.size(); ++i) push(i);
        while (!queue.empty()) {
            auto c = queue.top(); queue.pop();
            auto& l = nodes[c.left]; auto& r = nodes[c.right];
            if (!l.live || !r.live || l.next != c.right || l.version != c.lv || r.version != c.rv) continue;
            l.id = c.result; l.next = r.next; ++l.version; r.live = false;
            if (r.next != none) nodes[r.next].prev = c.left;
            push(l.prev); push(c.left);
        }
        for (size_t i=0; i != none && i<nodes.size(); i=nodes[i].next) output.push_back(nodes[i].id);
    }
    return {};
}
} // namespace tokenizer_detail
