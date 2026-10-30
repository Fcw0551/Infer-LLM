#include "../include/scheduler.hpp"

// 每次推理传一张全新的图进来
std::expected<void, std::error_code> Scheduler::execute(Graph& g){
    std::cout<<"execute"<<std::endl;
    // 切分子图 
    _subgraphs.clear();
    if (auto r = split_graph(g); !r)
        return r;
    // 生成规划图
    if (auto r = build_activation_graph(g); !r)
        return r;
    // 规划图，生命周期分析 activation达到复用结果，减少内存占用,分配内存
    if (auto r = alloc_graph(_activation_graph); !r)
        return r;
    // 执行子图：先把前一阶段算出来、但不在本设备上的边界张量搬过来，再算本子图
    for (auto &sub : _subgraphs){
        for (auto &c : sub.copies){
            // TODO，没有做同步
            if (auto r = copy_tensor(c.first, c.second); !r)
                return r;
        }
        if (auto r = sub.backend->compute(sub.graph); !r)
            return r;
    }

    return {};
}



// 划分子图
std::expected<void, std::error_code> Scheduler::split_graph(Graph &g){
    _subgraphs.clear();

    // 五步走，从被权重钉住的算子出发
    // 权重永远不搬，跨设备的只能是激活。
    
    // ----- 先找权重算子，权重在加载的时候已经确定了backend了，根据权重算子确定其消费者算子的backend--------
    //       有权重输入的算子优先留在权重所在 backend；权重 backend 不支持该算子时保持未赋值，
    //       交给 pass2 扩张 / pass3 未定分支 / pass4 兜底
    for(auto&node:g.nodes()){
        // 已经有 backend 的节点不再重新锚定
        if(node->backend){
            continue;
        }
        // ROPE / FLASH_ATTN_EXT 不按权重选 backend（freqs/sinks 太小，按它选会选错设备）
        if(node->op==OperationType::GGML_OP_ROPE || node->op==OperationType::GGML_OP_FLASH_ATTN_EXT){
            continue;
        }
        // 只拿第一个权重 src 当锚点，命中后立即结束该节点的 src 扫描
        for(size_t i=0;i<TENSOR_MAX_SRC&&node->src[i];i++){
            Tensor* s=node->src[i];

            if((static_cast<uint8_t>(s->tensorRole)&static_cast<uint8_t>(TensorRole::TENSOR_ROLE_WEIGHT))==0){
                continue;
            }
            // 权重加载时已经 bind 过 backend   为空说明加载/绑定漏了，属于参数错误
            if(!s->backend){
                return std::unexpected(std::make_error_code(std::errc::invalid_argument));
            }
            Backend* b=s->backend;
            if(b->supports_op(node->op,*node)){
                node->backend=b;
                node->device=b->get_info().type;
                node->pinned=true;          // 被权重钉住，后面 pass3 的升级逻辑不能动它
            }
            // else: 不报错、保持未赋值 （这里可能是当前backend不支持算子）
            //       由后续 pass 兜底（代价是权重可能需要拷贝）
            break;                      
        }
    }

    // ------- 向前/后扩展 扩张邻居 以高速设备为主，cpu设备最后（核心是避免断层）----------
    // 沿着拓扑序扫描计算图，遇到已经确定 Backend 的节点，就记住它，之后遇到尚未确定 Backend 的节点，就尝试让它跟随这个 Backend。
    // 不再要求“所有已赋值 src 的 backend 一致”，改成已赋值节点沿拓扑序向相邻未赋值节点扩张。
    // 固定四轮：GPU 向下 -> GPU 向上 -> 全部向下 -> 全部向上。
    // 前两轮跳过 host（CPU 优先级最低，不让它抢走 GPU 段），后两轮 CPU 也参与。
    // 扩张时 backend 不支持该算子就留空，交给 pass3 的未定分支 / pass4 兜底。
    auto expand = [&](bool forward, bool skip_host)
    {
        Backend *cur = nullptr;        // 当前扩张中的 backend
        const auto &nodes = g.nodes(); 
        for (size_t k = 0; k < nodes.size(); k++)
        {
            Tensor *node = nodes[forward ? k : nodes.size() - 1 - k]; // forward=false 时按逆拓扑序向上扩张
            if (node->backend){
                // 遇到已赋值节点就切换当前 backend；前两轮碰到 host 则断开（CPU 不向外扩张）
                cur = (skip_host && node->backend->is_host()) ? nullptr : node->backend;
            }
            else if (cur && cur->supports_op(node->op, *node))
            {
                // 未赋值且当前 backend 支持 -> 跟随扩张
                node->backend = cur;
                node->device = cur->get_info().type;
            }
            // 不支持时 cur 不重置，下一个节点继续尝试同一个 backend
        }
    };
    expand(true, true);   // pass2-1: GPU down（跳过 host）
    expand(false, true);  // pass2-2: GPU up（跳过 host）
    expand(true, false);  // pass2-3: rest down（host 参与）
    expand(false, false); // pass2-4: rest up（host 参与）

    // ------- 解决第一步和第二步遗留下来的：对于依旧未知的节点，主动遍历可用 Backend，寻找一个支持该算子的设备--------
    // 情况一：整张图没有任何backend锚点，一开始就没有显式指定设备，那么pass1 2 所有的backend都为null
    // 情况二：有些还未分配backend（所有backend算子不支持/ 前后夹着gpu，但是中间只支持cpu，导致枚举不了cpu的）
    // 1：未赋值节点：在“支持该算子”的 backend 里，选它的src大部分在哪个backend的
    //    平局取 _backends 里靠前的（约定越靠前优先级越高）
    //
    // 2：已赋值节点：被权重 pin 住的不动；否则尝试升级到更靠前（更高优先级）且支持该算子的 backend(在分配正确的前提下尽可能的安排更优)
    //    前提是所有 src 都已经在它上面（不会引入额外拷贝）。
    for(auto& node:g.nodes()){
        if(!node->backend){
            // 未赋值：选支持该算子、且选它的src大部分在哪个backend的
            Backend* best=nullptr;
            int best_cnt=-1;                                      //等会至少要让一个候选上
            for(Backend* b:_backends){
                if(!b->supports_op(node->op,*node)) continue;     // 候选必须支持该算子
                int cnt=0;
                for(size_t i=0;i<TENSOR_MAX_SRC&&node->src[i];i++){
                    if(node->src[i]->backend==b) ++cnt;
                }
                if(cnt>best_cnt){                                 // 严格大于 -> 平局保留更靠前的 backend
                    best_cnt=cnt;
                    best=b;
                }
            }
            if(best){
                node->backend=best;
                node->device=best->get_info().type;
            }
            // 一个支持该算子的 backend 都没有
            else{
                return std::unexpected(std::make_error_code(std::errc::not_supported));
            }
        }else{
            // 权重锚定的节点不允许被升级（pinned）
            if(node->pinned) continue;
            // 只在当前 backend 之前（更高优先级）的候选里找升级目标
            for(Backend* b:_backends){
                if(b==node->backend) break;                       // 走到当前位置就停，不考虑更低的
                if(!b->supports_op(node->op,*node)) continue;
                bool ok=true;
                for(size_t i=0;i<TENSOR_MAX_SRC&&node->src[i];i++){
                    Tensor* s=node->src[i];
                    // 要求 src 已赋值且就在 b 上，否则升级会引入额外拷贝（相当于在src里面挑一个更好的backend）
                    // TODO：没有考虑消费者，后续可以记录消费者
                    if (!s->backend || s->backend != b){
                        ok = false;
                        break;
                    }
                }
                if(ok){
                    node->backend=b;
                    node->device=b->get_info().type;
                    break;
                }
            }
        }
    }

    // ---------解决 leaf 的 backend：前面的 pass 只遍历了 nodes，graph input / rope 表等 leaf 还没赋值 ---------
    // 未赋值的 src 统一跟随消费者的 backend：
    //   - graph input 跟随消费者，而 embedding 节点已经被 embedding 权重 pin 住，
    //     所以 input 直接落在 embedding 权重所在的 backend，不会出现“随便挑一个 host”导致两边分离
    //   - 权重 src 不动：权重数据在加载时已固定，跨设备由 pass5 插拷贝
    // （节点这里不用再兜底：pass3 的未定分支已经保证每个节点要么有 backend，要么在上面直接报错返回）
    for(auto& node:g.nodes()){
        Backend* b=node->backend;
        for(size_t i=0;i<TENSOR_MAX_SRC&&node->src[i];i++){
            Tensor* s=node->src[i];
            if(s->backend) continue;                                    // 已赋值的 src 不动
            // 权重永远不搬（正常情况下加载时已经有 backend，这里防一手空指针）
            if((static_cast<uint8_t>(s->tensorRole)&static_cast<uint8_t>(TensorRole::TENSOR_ROLE_WEIGHT))!=0){
                continue;
            }
            s->backend=b;
            s->device=b->get_info().type;
        }
    }

    //---- pass5: 按连续的 backend 切分子图 + 在边界插拷贝 ------TODO 权重结点没有区分，可以做区分，然后只拷贝一次常驻
    // 1：拓扑序里相邻且 backend 相同的节点归为一个子图
    // 2：子图内某个节点的 src 不在本子图的 backend 上：在本子图 backend 上建一个拷贝张量，
    //    把 node->src[j] 改指向拷贝张量；同一子图内同一个 src 只建一份
    // 3：拷贝张量通过 g.adopt(unique_ptr) 交给总图持有：
    //    生命周期跟总图走，_subgraphs.clear() 不会释放它，node->src 不会悬空；
    //    它只被 Scheduler 读写，不参与拓扑排序，也不会被 backend 当节点执行
    // 4：执行顺序：execute() 里先做本子图的 copies，再 compute 本子图
    //    由于子图按拓扑序切分，拷贝的源一定在更早的子图里执行完 TODO
    //
    // 注意：这里会改写 node->src。同一张图重复 split 不会悬空（拷贝归总图），
    //       但上一轮的拷贝会被当成“已就位”直接复用、不会重新执行拷贝，
    //       所以仍然约定“每次推理传一张全新的图”（见 execute 开头）
    const auto &nodes = g.nodes();
    if (nodes.empty()) {
        // 有输出、但拓扑和 leafs 都是空的 => Graph::compute() 还没建拓扑序。
        // 不能当成成功，否则 execute 会静默地什么都不执行
        if (!g.outputs().empty() && g.leafs().empty()) {
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        }
        return {};
    }
    // pass1~4 之后每个节点都必须有 backend，否则后面没法定拷贝
    for (Tensor *node : nodes) {
        if (!node->backend) {
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        }
    }

    // 1：切分子图（连续同 backend 的节点为一段）
    size_t start = 0;
    for (size_t i = 1; i <= nodes.size(); i++) {
        if (i < nodes.size() && nodes[i]->backend == nodes[start]->backend) {
            continue;
        }
        Subgraph sub;
        sub.backend = nodes[start]->backend;
        for (size_t k = start; k < i; k++) {
            sub.graph.add_node_ref(nodes[k]);
        }
        _subgraphs.push_back(std::move(sub));
        start = i;
    }

    // 2：节点 -> 所属子图下标，用来填每个子图的 outputs（记录tensor属于哪个子图）
    std::unordered_map<Tensor *, size_t> node_sub;
    for (size_t s = 0; s < _subgraphs.size(); s++) {
        for (Tensor *node : _subgraphs[s].graph.nodes()) {
            node_sub[node] = s;
        }
    }

    // 3：找跨设备边界，建拷贝张量并改写 src
    std::vector<std::unordered_set<Tensor *>> seen_output(_subgraphs.size());// 用于记录某个生产者子图的 Tensor 是否已经被登记为跨子图输出，防止重复添加
    for (size_t s = 0; s < _subgraphs.size(); s++) {
        Subgraph &sub = _subgraphs[s];
        std::unordered_map<Tensor *, Tensor *> copy_map;   // 本子图内 src -> 拷贝，去重

        // 遍历当前子图的所有node，找其src输入结点是否是在别的子图里
        for (Tensor *node : sub.graph.nodes()) {
            for (size_t j = 0; j < TENSOR_MAX_SRC && node->src[j]; j++) {
                Tensor *src = node->src[j];
                if (!src->backend) {
                    return std::unexpected(std::make_error_code(std::errc::invalid_argument));
                }

                // 源是更早子图里的节点 -> 生产者子图要把这个张量保活到本轮结束

                // 找当前的输入结点，它在别的子图里面，但是不属于当前子图
                auto pit = node_sub.find(src);
                if (pit != node_sub.end() && pit->second != s) {
                    // 如果是跨结点，则尝试插入，如果插入失败，说明之前没有，如果插入成功，说明之前存在，防止重复添加
                    if (seen_output[pit->second].insert(src).second) {
                        // 插入成功,则往子图的outputs输出添加标记
                        _subgraphs[pit->second].outputs.push_back(src);
                    }
                }

                // 已经在同一设备上，直接读，不用拷
                if (src->backend == sub.backend) {
                    continue;
                }

                // ---- 没有在同一设备当中 -----
                Tensor *cpy = nullptr;
                auto it = copy_map.find(src);
                // 检查是不是拷贝过了
                if (it != copy_map.end()){
                    cpy = it->second;
                }
                else{
                    // 没有拷贝过，执行拷贝
                    auto t = std::make_unique<Tensor>(src->name + "_##copy##", src->dataType,
                        std::initializer_list<int64_t>{src->dims[0], src->dims[1], src->dims[2], src->dims[3]});
                    t->dims = src->dims; // 拷贝目标和源同 shape / 同布局
                    t->strides = src->strides;
                    t->tensorRole = TensorRole::TENSOR_ROLE_ACTIVATION;
                    t->backend = sub.backend;
                    t->device = sub.backend->get_info().type;
                    t->src[0] = src;
                    cpy = g.adopt(std::move(t));   // 所有权归总图，不参与拓扑
                    copy_map.emplace(src, cpy);
                    sub.copies.emplace_back(src, cpy);
                    sub.inputs.push_back(src);
                }
                // 本子图的算子改吃本设备上的拷贝
                node->src[j] = cpy;
            }
        }
    }
    // 图自身的输出没有消费者，不会被上面的 src 扫描收集到，这里单独补
    for (Tensor *out : g.outputs()){
        auto it = node_sub.find(out);
        if (it != node_sub.end() && seen_output[it->second].insert(out).second){
            _subgraphs[it->second].outputs.push_back(out);
        }
    }

    return {};
}

// 跨设备拷贝：先试 dst 设备的 P2P 高速路径，不行再退回 host 中转
std::expected<void, std::error_code> Scheduler::copy_tensor(Tensor *src, Tensor *dst){
    if (!src || !dst || !src->backend || !dst->backend) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (src->backend == dst->backend) {
        return {};
    }
    // 拷贝张量的 data 由 alloc_graph 分配
    if (!src->data || !dst->data) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    // 高速路径：由 dst 设备判断能不能不用 host 直接搬
    auto result=dst->backend->copy_tensor_high(src, dst);
    if (!result) {
        return std::unexpected(result.error());
    }
    else{
        if(*result){
            return {};
        }
    }

    // 兜底：src 设备 -> host -> dst 设备
    std::vector<char> host(src->byte_size());
    if (auto r = src->backend->copy_tensor_get(src, host.data(), 0, host.size()); !r) {
        return r;
    }
    if (auto r = dst->backend->copy_tensor_set(dst, host.data(), 0, host.size()); !r) {
        return r;
    }
    return {};
}



// 申请那些还没有data的激活值
std::expected<void, std::error_code> Scheduler::alloc_graph(Graph &g){
    // split规划好计算图之后，要传给activationManager 做activation的复用
   
    // 根据分配图规划 Tensor 的内存布局
    auto result = _ac.plan(_activation_graph);
    if (!result) {
        return std::unexpected(result.error());
    }

    // 确保底层 Activation Buffer 容量足够
    result = _ac.allocate();
    if (!result) {
        return std::unexpected(result.error());
    }

    // 根据规划结果绑定 tensor->data
    result = _ac.bind();
    if (!result) {
        return std::unexpected(result.error());
    }

    return {};
}




// --------------------------------- helper -----------------------------

// 初始化内存容量 TODO如果权重过大，内存小，但是有时候也可以通过磁盘的换入换出跑起来
std::expected<std::unordered_map<Backend*, size_t>,std::error_code> Scheduler::init_memory_budgets(Graph &g){
    std::unordered_map<Backend *, size_t> budgets;
    for (Backend *b : _backends){
        budgets[b] = static_cast<size_t>(b->get_info().memory_bytes * MEMORY_FACTOR);
    }

    // 扣除常驻权重：权重在加载时已经占住了这块内存
    for (Tensor* leaf : g.leafs()) {
        const uint8_t role = static_cast<uint8_t>(leaf->tensorRole);
        if ((role & static_cast<uint8_t>(TensorRole::TENSOR_ROLE_WEIGHT)) == 0) {
            continue;
        }
        if (!leaf->backend) {
            continue;
        }
        auto it = budgets.find(leaf->backend);
        if (it == budgets.end()) {
            continue;
        }
        const size_t bytes = leaf->byte_size();
        if (bytes > it->second){ 
            // 权重本身就超了
            return std::unexpected(std::make_error_code(std::errc::not_enough_memory));
        }
        it->second -= bytes;
    }
    return budgets;
}

std::expected<void, std::error_code> Scheduler::build_activation_graph(Graph &graph){
    // 遍历每一张子图，然后构建出来一张完整的plan graph
    _activation_graph.clear();
    for (auto &sub : _subgraphs){
        // 先引入拷贝结点，因为拷贝结点是依赖前面的子图的
        for (auto &copy : sub.copies){
            _activation_graph.add_node_ref(copy.second);
        }
        // 后引入计算结点，因为计算结点依赖拷贝结点
        for (Tensor *node : sub.graph.nodes()){
            _activation_graph.add_node_ref(node);
        }
        for (Tensor *output : graph.outputs())
            _activation_graph.mark_output(output);
    }
    return {};
}