#pragma once
#include <climits>
#include <cstddef>
#include <expected>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

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
    // 把计算图导出成文件：自己从 outputs()/inputs() 出发沿 src[] 做 DFS，
    // 不依赖 Graph::build_topo_order（那个还没实现，_nodes/_leafs 目前是空的）。
    //
    // layer_lo / layer_hi 是"只画哪几层"的闭区间，用来缩小图：
    //   dump(g, path)            → 默认 [0, INT_MAX]，整张图，行为和加这个参数之前一致
    //   dump(g, path, 0, 1)      → 只画第 0、1 层
    // 层的判定靠解析张量名字（Tensor::layer 只有权重被赋过值，中间张量是垃圾值，不能用）：
    //   "blk.<N>.xxx" → N     "<前缀><N>" → N     解析不出（如 logits/rope_theta）→ 全局节点，永远保留
    static std::expected<void, std::error_code> dump(const Graph& g, const std::string& path,
                                                     int layer_lo = 0, int layer_hi = INT_MAX);

    size_t node_count() const noexcept { return _nodes.size(); }
    size_t edge_count() const noexcept { return _edges.size(); }

private:
    std::string _name;                 // digraph 的名字
    std::vector<std::string> _nodes;   // 每个点一行，不含 digraph{} 外壳
    std::vector<std::string> _edges;   // 每条边一行
};
