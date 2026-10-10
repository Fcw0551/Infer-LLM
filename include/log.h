#pragma once
#include <climits>
#include <cstddef>
#include <expected>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

// dot 图的落地目录：和 include/ src/ 同级的 models_dot/。
// 优先用 CMake 传进来的绝对路径（见 CMakeLists 的 INFERLLM_MODELS_DOT_DIR），
// 没定义就退回相对路径 "models_dot"（相对当前工作目录）。
// 用 DotWriter::dot_path("qwen3_graph.dot") 拼完整路径；目录不存在 save() 会自动建。
#ifndef INFERLLM_MODELS_DOT_DIR
#define INFERLLM_MODELS_DOT_DIR "models_dot"
#endif

class Graph;   // 前置声明：log.h 不依赖 graph.hpp，谁用谁自己 include

// 把一张图写成 Graphviz DOT 格式。
// 生成的文件丢到 https://dreampuf.github.io/GraphvizOnline/ 看，
// 或本地 `dot -Tsvg graph.dot -o graph.svg`。
//
// 两种用法：
//   1) 高层：DotWriter::dump(g, "qwen3_graph.dot");          // Graph → 文件，一行搞定
//   2) 底层：DotWriter w("G");
//            size_t a = w.add_node("input");
//            size_t b = w.add_node("double");
//            w.add_edge(a, b);
//            w.save("tiny.dot");
class DotWriter {
public:
    explicit DotWriter(std::string graph_name = "G");

    // ---------- 底层 API ----------
    // 加一个点，返回它的编号（从 0 开始），供 add_edge 用。
    // label 里含 " \ 换行 也不用管，内部会转义；换行会变成 DOT 里的换行。
    size_t add_node(std::string_view label, std::string_view attrs = {});
    // 加一条有向边 from → to。编号越界则忽略（返回 void，不崩）。
    void   add_edge(size_t from, size_t to, std::string_view attrs = {});

    // ---------- 落盘 ----------
    std::expected<void, std::error_code> save(const std::string& path) const;

    // ---------- 高层 API ----------
    // 把计算图导出成文件。点直接取自 Graph 建好的拓扑序（nodes() + leafs()）：
    // 叶子是 build_topo_order 按 op == GGML_OP_NONE 分好的，这里不再自己判断叶子。
    //
    // 前置条件：调用前必须先跑过 Graph::compute()（拓扑序在那里建），
    // 否则 nodes()/leafs() 还是空的，dump 会返回 invalid_argument。
    //
    // layer_lo / layer_hi 是"只画哪几层"的闭区间，用来缩小图：
    //   dump(g, path)            → 默认 [0, INT_MAX]，整张图
    //   dump(g, path, 0, 1)      → 只画第 0、1 层
    // 层的判定靠解析张量名字（Tensor::layer 只有权重被赋过值，中间张量是垃圾值，不能用）：
    //   "blk.<N>.xxx" → N     "<前缀><N>" → N     解析不出（如 logits/rope_theta）→ 全局节点，永远保留
    static std::expected<void, std::error_code> dump(const Graph& g, const std::string& path,
                                                     int layer_lo = 0, int layer_hi = INT_MAX);

    // 拼出 dot 文件在 models_dot/ 下的完整路径，调用点直接用它，别自己写相对路径——
    // 相对路径会跟着"在哪个目录跑"乱跑。目录不存在也没关系，save() 会建。
    static std::string dot_path(std::string_view filename);

    size_t node_count() const noexcept { return _nodes.size(); }
    size_t edge_count() const noexcept { return _edges.size(); }

private:
    std::string _name;                 // digraph 的名字
    std::vector<std::string> _nodes;   // 每个点一行，不含 digraph{} 外壳
    std::vector<std::string> _edges;   // 每条边一行
};
