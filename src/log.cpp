// log.cpp —— 计算图可视化：把图导出成 Graphviz DOT
#include "../include/log.h"
#include "../include/graph.hpp"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

// 转义成能安全放进 DOT 双引号里的形式；换行变成 DOT 的 \n（渲染时换行）
std::string escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// OperationType → 名字。常用的手写，其余统一打成 OP_<数值>，
// 这样将来加了新算子也不会打错，只是名字不好看而已。
std::string op_name(OperationType op) {
    switch (op) {
        case OperationType::GGML_OP_NONE:           return "NONE";
        case OperationType::GGML_OP_DUP:            return "DUP";
        case OperationType::GGML_OP_ADD:            return "ADD";
        case OperationType::GGML_OP_ADD1:           return "ADD1";
        case OperationType::GGML_OP_SUB:            return "SUB";
        case OperationType::GGML_OP_MUL:            return "MUL";
        case OperationType::GGML_OP_DIV:            return "DIV";
        case OperationType::GGML_OP_SUM:            return "SUM";
        case OperationType::GGML_OP_MEAN:           return "MEAN";
        case OperationType::GGML_OP_CONCAT:         return "CONCAT";
        case OperationType::GGML_OP_NORM:           return "NORM";
        case OperationType::GGML_OP_RMS_NORM:       return "RMS_NORM";
        case OperationType::GGML_OP_GROUP_NORM:     return "GROUP_NORM";
        case OperationType::GGML_OP_L2_NORM:        return "L2_NORM";
        case OperationType::GGML_OP_MUL_MAT:        return "MUL_MAT";
        case OperationType::GGML_OP_MUL_MAT_ID:     return "MUL_MAT_ID";
        case OperationType::GGML_OP_OUT_PROD:       return "OUT_PROD";
        case OperationType::GGML_OP_SCALE:          return "SCALE";
        case OperationType::GGML_OP_CPY:            return "CPY";
        case OperationType::GGML_OP_CONT:           return "CONT";
        case OperationType::GGML_OP_RESHAPE:        return "RESHAPE";
        case OperationType::GGML_OP_VIEW:           return "VIEW";
        case OperationType::GGML_OP_PERMUTE:        return "PERMUTE";
        case OperationType::GGML_OP_TRANSPOSE:      return "TRANSPOSE";
        case OperationType::GGML_OP_GET_ROWS:       return "GET_ROWS";
        case OperationType::GGML_OP_DIAG_MASK_INF:  return "DIAG_MASK_INF";
        case OperationType::GGML_OP_SOFT_MAX:       return "SOFT_MAX";
        case OperationType::GGML_OP_ROPE:           return "ROPE";
        case OperationType::GGML_OP_ARGSORT:        return "ARGSORT";
        case OperationType::GGML_OP_FLASH_ATTN_EXT: return "FLASH_ATTN_EXT";
        case OperationType::GGML_OP_UNARY:          return "UNARY";
        case OperationType::GGML_OP_GLU:            return "GLU";
        default: return "OP_" + std::to_string(static_cast<uint32_t>(op));
    }
}

// 形状。Tensor 没存 n_dims，靠"第二维是 0 就不打"来区分 1 维 / 2 维 ——
// 这依赖 TensorInfo::_dims 是零初始化的（见 gguf_parser.hpp:39 的 _dims{}），
// 没写过的维度是 0 而不是垃圾值。否则 norm 权重会打出 [1024 x 3208777248] 这种鬼东西。
//   1维的 norm 权重 → [1024]；2维的权重 → [1024 x 151936]；input_tokens → [4]
std::string shape_str(const Tensor* t) {
    std::string s = "[" + std::to_string(t->dims[0]);
    if (t->dims[1] != 0) s += " x " + std::to_string(t->dims[1]);
    return s + "]";
}

// 从张量名字里解析"第几层"；解析不出来返回 -1，表示全局节点（过滤时永远保留）。
// 注意：不能读 Tensor::layer —— 那个字段只有权重被赋过值（见 model.cpp 的 create_tensor），
// OpFactory 建的中间张量是未初始化的垃圾值。所以只能认命名习惯（见 Qwen3.cpp 的 build_graph）：
//   权重   "blk.<N>.xxx"        → N
//   中间量 "<任意前缀><N>"       → N    （q_proj0 / result_out_27 / ffn_gate_silu_3 …）
//   全局   "logits" / "rope_theta" / "token_embd.weight" … 结尾不是数字 → -1
int tensor_layer(std::string_view name) {
    // 1) blk.<N>.…
    if (name.size() > 4 && name.compare(0, 4, "blk.") == 0) {
        size_t i = 4;
        if (i >= name.size() || !std::isdigit(static_cast<unsigned char>(name[i]))) return -1;
        int v = 0;
        while (i < name.size() && std::isdigit(static_cast<unsigned char>(name[i])))
            v = v * 10 + (name[i++] - '0');
        return (i < name.size() && name[i] == '.') ? v : -1;
    }
    // 2) 结尾的一串数字
    size_t i = name.size();
    while (i > 0 && std::isdigit(static_cast<unsigned char>(name[i - 1]))) --i;
    if (i == name.size() || i == 0) return -1;   // 结尾不是数字 / 整个名字都是数字 → 当全局
    int v = 0;
    for (size_t k = i; k < name.size(); ++k) v = v * 10 + (name[k] - '0');
    return v;
}

} // namespace

// =========================== DotWriter ===========================

DotWriter::DotWriter(std::string graph_name) : _name(std::move(graph_name)) {}

size_t DotWriter::add_node(std::string_view label, std::string_view attrs) {
    size_t nid = _nodes.size();
    std::string line = "n" + std::to_string(nid) + " [label=\"" + escape(label) + "\"";
    if (!attrs.empty()) line += ", " + std::string(attrs);
    line += "];";
    _nodes.push_back(std::move(line));
    return nid;
}

void DotWriter::add_edge(size_t from, size_t to, std::string_view attrs) {
    if (from >= _nodes.size() || to >= _nodes.size()) return;   // 越界忽略，别崩
    std::string line = "n" + std::to_string(from) + " -> n" + std::to_string(to);
    if (!attrs.empty()) line += " [" + std::string(attrs) + "]";
    line += ";";
    _edges.push_back(std::move(line));
}

std::expected<void, std::error_code> DotWriter::save(const std::string& path) const {
    // 目录（models_dot/ 之类）不存在就先建出来，省得调用方自己 mkdir。
    // 建不出来不在这里报错，交给下面的 ofstream 去报，错误信息更准。
    const std::filesystem::path p(path);
    if (p.has_parent_path() && !p.parent_path().empty()) {
        std::error_code ignored;
        std::filesystem::create_directories(p.parent_path(), ignored);
    }

    std::ofstream f(path, std::ios::out | std::ios::trunc);
    if (!f) return std::unexpected(std::make_error_code(std::errc::io_error));

    f << "digraph \"" << escape(_name) << "\" {\n";
    f << "  rankdir=TB;\n";
    f << "  node [shape=box, style=filled, fontname=\"Helvetica\"];\n";
    for (const auto& n : _nodes) f << "  " << n << "\n";
    for (const auto& e : _edges) f << "  " << e << "\n";
    f << "}\n";

    if (!f.good()) return std::unexpected(std::make_error_code(std::errc::io_error));
    return {};
}

std::string DotWriter::dot_path(std::string_view filename) {
    // INFERLLM_MODELS_DOT_DIR 由 CMake 传进来（绝对路径）；没定义就用 log.h 里的 "models_dot"
    return (std::filesystem::path(INFERLLM_MODELS_DOT_DIR) / filename).string();
}

std::expected<void, std::error_code> DotWriter::dump(const Graph& g, const std::string& path,
                                                     int layer_lo, int layer_hi) {
    // 点直接取自 Graph 建好的拓扑序：叶子/算子已经由 build_topo_order 分好类了，
    // 这里不再自己判断谁是叶子。所以调用前必须先跑过 Graph::compute()。
    const std::vector<Tensor*>& leafs = g.leafs();
    const std::vector<Tensor*>& nodes = g.nodes();
    if (leafs.empty() && nodes.empty()) {
        // 拓扑序还没建：图没连、没 mark_output，或者压根没调 compute()。
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    // 输入/输出集合，只用来着色。
    std::unordered_set<const Tensor*> is_input(g.inputs().begin(), g.inputs().end());
    std::unordered_set<const Tensor*> is_output(g.outputs().begin(), g.outputs().end());

    // 按层过滤。全局节点（解析不出层号）永远保留，其余要求层号落在 [lo, hi]。
    auto keep = [&](const Tensor* t) {
        const int L = tensor_layer(t->name);
        return L < 0 || (L >= layer_lo && L <= layer_hi);
    };

    DotWriter w("Graph");
    std::unordered_map<const Tensor*, size_t> ids;

    // 建点。leaf 这个参数直接来自拓扑序的分类，不用再猜。
    auto emit_node = [&](const Tensor* t, bool leaf) {
        if (!keep(t)) return;
        if (ids.count(t)) return;   // 防御：同一个张量只画一次

        // 配色：输出红 > 输入绿 > 叶子灰 > 中间节点蓝
        std::string color = leaf ? "#EEEEEE" : "#BBDEFB";
        if (is_input.count(t))  color = "#C8E6C9";
        if (is_output.count(t)) color = "#FFCDD2";

        std::string label = t->name + "\n" +
                            (leaf ? std::string("leaf") : op_name(t->op)) + "\n" +
                            shape_str(t) + " " + ggml_type_name(t->dataType);

        std::string attrs = "fillcolor=\"" + color + "\"";
        if (leaf) attrs += ", shape=ellipse";

        ids.emplace(t, w.add_node(label, attrs));
    };

    for (const Tensor* t : leafs) emit_node(t, true);
    for (const Tensor* t : nodes) emit_node(t, false);

    // 建边：只画两端都留下来的（被层过滤掉的邻居会让这条边悬空，直接丢弃）。
    // 拓扑序保证 src 一定在 dst 之前被访问过，所以这里不需要再排一次序。
    auto emit_edges = [&](const Tensor* dst) {
        const auto b = ids.find(dst);
        if (b == ids.end()) return;
        for (const Tensor* s : dst->src) {
            if (!s) continue;
            const auto a = ids.find(s);
            if (a == ids.end()) continue;
            w.add_edge(a->second, b->second);
        }
    };

    for (const Tensor* t : leafs) emit_edges(t);
    for (const Tensor* t : nodes) emit_edges(t);

    return w.save(path);
}
