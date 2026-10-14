#include "../../include/backends/cpu_backend.hpp"

// ------ 线程池 -------
void ThreadPool::init(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    shutdown();
    _n_threads = n_threads;
    _stop = false;
    _epoch = 0;

    for (int i = 0; i < n_threads; ++i) {
        _threads.emplace_back(&ThreadPool::worker_loop, this, i);
    }
}

void ThreadPool::shutdown() {
    {
        std::lock_guard<std::mutex> lk(_mtx);
        _stop = true;
    }
    _cv_start.notify_all();
    for (auto& t : _threads) {
        if (t.joinable()) t.join();
    }
    _threads.clear();
    _n_threads = 0;
}

void ThreadPool::worker_loop(int tid) {
    int last_epoch = 0;
    while (true) {
        // 1. 等待新任务
        {
            std::unique_lock<std::mutex> lk(_mtx);
            _cv_start.wait(lk, [&] {
                return _stop || _epoch != last_epoch;
            });
            if (_stop) return;
            last_epoch = _epoch;
        }

        // 2. 并行处理任务
        while (true) {
            int64_t i = _next.fetch_add(1, std::memory_order_relaxed);
            if (i >= _n) break;
            _fn(i);
        }

        // 3. 报告完成
        {
            std::lock_guard<std::mutex> lk(_mtx);
            _n_done++;
            if (_n_done == _n_threads) {
                _cv_done.notify_one();
            }
        }
    }
}

void ThreadPool::parallel_for(int64_t n, const std::function<void(int64_t)>& fn) {
    if (n <= 0) return;

    // 单线程：直接跑
    if (_n_threads <= 1) {
        for (int64_t i = 0; i < n; ++i) fn(i);
        return;
    }

    {
        std::lock_guard<std::mutex> lk(_mtx);
        _fn    = fn;
        _n     = n;
        _next  = 0;
        _n_done = 0;
        _epoch++;
    }
    _cv_start.notify_all();

    // 主线程也参与
    while (true) {
        int64_t i = _next.fetch_add(1, std::memory_order_relaxed);
        if (i >= n) break;
        fn(i);
    }

    // 等所有 worker 完成
    {
        std::unique_lock<std::mutex> lk(_mtx);
        _cv_done.wait(lk, [&] {
            return _n_done == _n_threads;
        });
    }
}





// cpu 读取total_ram
size_t get_total_memory() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}
//cpu 后端的注册函数
void register_cpu_backend(){
    BackendRegistry::instance().register_backend("cpu",
    []()->std::vector<DeviceInfo> { return {DeviceInfo{"cpu", DeviceType::CPU, get_total_memory()}}; },   // probe
    [](int device_id)->std::unique_ptr<Backend> { return std::make_unique<CPUBackend>(device_id); } // create
    );
}


// CPUBackend
// 跨设备传输  copy结点
// 从 host 的内存数据 写入这个 tensor 所在的设备的buffer
std::expected<void, std::error_code> CPUBackend::copy_tenosr_set(Tensor *tensor, const void *data, size_t offset, size_t size) {
    assert(tensor->device==DeviceType::CPU);
    if(data==nullptr){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (offset + bytes > tensor->byte_size()) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    std::memcpy(static_cast<char*>(tensor->data) + offset, data, size);
    return {};
}
// 从这个 tensor 所在的设备 buffer，读数据到 host 内存。
std::expected<void, std::error_code> CPUBackend::copy_tenosr_get(const Tensor *tensor, void *data, size_t offset, size_t size) {
    assert(tensor->device==DeviceType::CPU);

    if (data == nullptr){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (tensor->data == nullptr) {
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }
    if (offset + bytes > tensor->byte_size()){
        return std::unexpected(std::make_error_code(std::errc::invalid_argument));
    }

    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, bytes);
    return {};
}


// cpu初始化
void CPUBackend::cpu_init(){
    
    // 环境变量
    if (const char *env = std::getenv("INFERLLM_THREADS")){
        _n_threads=std::stoi(env);              
    }
    else{
        _n_threads=0;                           //交给下一步
    }

    // 自动检测
    if (_n_threads <= 0)
    {
        unsigned int hw = std::thread::hardware_concurrency();  //cpu逻辑核心数量
        if(!hw){
            hw=1;
        }
        // 安排策略 （TODO,实际可能要根据物理核心 和 逻辑核心进行合理安排）
        // 这里先做一个简单的0.5倍的逻辑核心
        // 要测试，如果某个线程数量已经够跑，够打满内存带宽，可能就不需要再多的线程，导致缓存失效，上下文切换--> 性能下降
        _n_threads=static_cast<int>(hw*0.5);
    }

    // 兜底：至少 1 线程
    if (_n_threads < 1)
        _n_threads = 1;

    std::cout << "[cpu] default threads = " << _n_threads << "\n";

    // 初始化线程池
    _threadPool.init(_n_threads);
}


// 执行 scheduler传子图给每个后端执行
std::expected<void, std::error_code> CPUBackend::compute(Graph &graph){
    // 上层scheduler做切分，已经弄好了子图
    // 子图的 nodes 已经是拓扑序——按序遍历
    for (Tensor* node : graph.nodes()) {
        // 叶子节点跳过
        if (node->op == OperationType::GGML_OP_NONE) {
            continue;
        }

        // 按 op 分派
        auto r = dispatch(node);
        if (!r) {
            std::cout << "[cpu] compute failed at node: "<< node->name << " op=" << (int)node->op << "\n";
            return r;
        }
    }
    return {};
}

std::expected<void, std::error_code> CPUBackend::dispatch(Tensor* node) {
    switch (node->op) {
    case OperationType::GGML_OP_ADD:            return compute_add(*node);
    case OperationType::GGML_OP_MUL:            return compute_mul(*node);
    case OperationType::GGML_OP_UNARY:
        {
            uint32_t op;
            std::memcpy(&op,&node->op_params[0],4);
            
            switch(static_cast<OperationType_unary_op>(op)){
                case OperationType_unary_op::GGML_UNARY_OP_SILU:    return compute_unary_silu(*node);
            }
           
        }
    case OperationType::GGML_OP_RMS_NORM:       return compute_rms_norm(*node);
    case OperationType::GGML_OP_GET_ROWS:       return compute_get_rows(*node);
    case OperationType::GGML_OP_MUL_MAT:        return compute_mul_mat(*node);
    case OperationType::GGML_OP_ROPE:           return compute_rope(*node);
    case OperationType::GGML_OP_FLASH_ATTN_EXT: return compute_flash_attn(*node);

    case OperationType::GGML_OP_NONE:
        return {};   // 叶子，跳过

    default:
        std::cerr << "[cpu] unsupported op: " << (int)node->op << "\n";
        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }
}


//-------cpu算子实现--------------
std::expected<void, std::error_code> CPUBackend::compute_get_rows(Tensor* dst){
    // TODO 先做 f32的 后续通过switch转化

    // 这个算子的本质是一个gather，一个索引+一个表
    // 索引去替换表的第二维就是结果
    Tensor* table= node->src[0];
    Tensor* indices= node->src[1];
    assert(src0&&src1);
    assert(src0->dims[0]==dst->dims[0]);
    assert(src1->dims[0]==dst->dims[1]);
    assert(src0->dims[2]==src1->dims[1]);
    assert(src0->dims[3]==src1->dims[2]);

    // llama.cpp是手动传入线程数量，每次精确计算
    // 这里采用的是持久化的线程池，直接把任务拆解传给线程池即可

    // F32
    //  table:   [nc, n_vocab, B1, B2]      ← 4 维
    //  indices: [n_idx, B1, B2]            ← 3 维
    //  输出:    [nc, n_idx, B1, B2]        ← 4 维

    // //  维度 
    // const int64_t nc  = table->dims[0];        // 每行元素数
    // const int64_t dims10 = indices->dims[0];     // 索引个数
    // const int64_t dims11 = indices->dims[1];     // 批次 1
    // const int64_t dims12 = indices->dims[2];     // 批次 2

    // //  步长（字节）
    // const int64_t strides10 = indices->strides[0];
    // const int64_t strides11 = indices->strides[1];
    // const int64_t strides12 = indices->strides[2];

    // const int64_t dims11 = table->strides[1];    // table 行步长
    // const int64_t dims12 = table->strides[2];
    // const int64_t dims13 = table->strides[3];

    // const int64_t strides1  = dst.strides[1];     // 输出行步长
    // const int64_t strides2  = dst.strides[2];
    // const int64_t strides3  = dst.strides[3];
    
    // const int64_t nc=table->dims[0];       //每行多少元素
    // const int64_t n_idx=indices->dims[0]; //索引的个数

    INFERLLM_TENSOR_BINARY_LOCALS              //宏展开
    
    const char *table_base = static_cast<const char *>(table->data);
    const char *indices_base = static_cast<const char *>(indices->data);
    char *dst_base = static_cast<char *>(dst->data);

    const int64_t nr=dims10*dims11*dims12;  //要遍历多少行

    _threadPool.parallel_for(nr, [&](int64_t i)
                            {
        // 每个线程拿到第 i 行
        // 先计算这个一维的第 i 行转化为坐标是（i10，i11，i12）
        const int64_t i12=i/(dims10*dims11);
        const int64_t i11=(i-i12*dims10*dims11)/dims10;
        const int64_t i10=(i-i12*dims10*dims11-i11*dims10);
        // 从 indices 读索引值 解引用四个字节
        int32_t id = *reinterpret_cast<const int32_t *>(indices_base + i10 * strides10 + i11 * strides11 + i12 * strides12);

        // 源地址：table 的 (id, i11, i12)
        const char *src = table_base + id * strides01 + i11 * strides02 + i12 * strides03;  

        // 目标地址：输出的 (i10, i11, i12)
        char *dst = dst_base + i10 * strides1 + i11 * strides2 + i12 * strides3;

        // 拷贝整行
        std::memcpy(dst, src, nc * sizeof(float)); }
    )
}


std::expected<void, std::error_code> CPUBackend::compute_rms_norm(Tensor* dst){
    // x            : [n_embd , n_tokens]
    // weight       : [n_embd,1,1,1]
    // rms_norm_i   : [n_embd , n_tokens]

    Tensor* src0 = dst->src[0];
    Tensor* src1 = dst->src[1];

    infer_are_same_shape(src0,dst);
    
    // 宏展开
    INFERLLM_TENSOR_BINARY_LOCALS

    float eps =0;
    std::memcpy(&eps,&dst->op_params[0],sizeof(eps));    //把eps拷贝出来
    
    const char *x_base = static_cast<const char *>(src0->data);
    const char *weight_base = static_cast<const char *>(src1->data);
    char *dst_base = static_cast<char *>(dst->data);

    //TODO：decode阶段只一个token的时候，cpu只用一个核心，这里主要是按照token数量来并行的
    //      后期可以改为token里面也分段，prefill阶段要根据核心数量动态调整，decode阶段可以一个token里面再细分
    //      还有关于SIMD向量化的优化可以后期考虑
    
    // 两次的for循环是必然的 rms比layer少一次for
    _threadPool.parallel_for(dims01, [&](int64_t i)
                            {
                                 // 求平方和  [n_embd , i]
                                const float* start=static_cast<const float *>(x_base+i*strides1);
                                const float* weight=static_cast<const float *>(weight_base);
                                float* dst=static_cast<float *>(dst_base+i*strides1);

                                double sum=0;           // TODO 这里用int 还是double 有没有考量呢

                                for(int index=0;index<dims00;index++){
                                    //sum+=std::pow(start[index],2);
                                    sum+=start[index] * start[index];
                                }
                                // 开根号  乘以γ weight
                                float scale=1.0f/std::sqrt(sum/dims00+eps);
                                for(int index=0;index<dims00;index++){
                                    dst[index]=start[index]*scale*weight[index];
                                } 
                            })
}
