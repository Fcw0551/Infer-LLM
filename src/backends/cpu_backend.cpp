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
    );
    return {};
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
                            });
    return {};
}



std::expected<void, std::error_code> compute_mul_mat(Tensor* dst){
    //w     : [n_embd, n_out]
    //x     : [n_embd, n_tokens] 
    //out   : [n_out, n_tokens]
    // x*w（转置）
    
    Tensor* src0 = dst->src[0];
    Tensor* src1 = dst->src[1];
    
    // 宏展开
    INFERLLM_TENSOR_BINARY_LOCALS

    assert(dims00 == dims10);
    assert(dst->ne[0] == dims01);
    assert(dst->ne[1] == dims11);
    const float *x_base = static_cast<const float *>(src1->data);
    const float *w_base = static_cast<const float *>(src0->data);
    float *dst_base = static_cast<float *>(dst->data);

    //TODO: 现在暂时按照输出的维度的每个位置来定一个任务
    _threadPool.parallel_for(dims01*dims11,[&](int64_t i){
        // 先按照输出的每个位置来定并行
        // i表示第i个任务 
        // out[a b]=x[a K]* w[K b]

        const int64_t token = i / dims01;
        const int64_t output  = i % dims01;
        
        //计算起始偏移量
        const float* x=x_base+token*dims10;
        const float* w=w_base+output*dims00;
        float sum=0.0f;
        for(int64_t k=0;k<dims00;k++){
            sum+=x[k]*w[k];
        }
        dst_base[i]=sum;
    });
    return {};
}

std::expected<void, std::error_code> CPUBackend::compute_rope(Tensor* dst){
    Tensor *src0 = dst->src[0];             // x  （q/k）
    Tensor *src1 = dst->src[1];            // inp_pos
    Tensor *src2 = dst->src[2];            // theta表（旋转角度表）
    INFERLLM_TENSOR_TERNARY_LOCALS

    infer_are_same_shape(src2,dst);

    // src0 (Q/K)   :  [head_dim, n_heads, n_tokens]
    // src1 (pos)   :  [n_tokens]
    // src2 (angle) :  [head_dim / 2 , n_ctx]
    // dst          :  [head_dim, n_heads, n_tokens]

    // rope θ表： 1/base_theta^(2.0*i/head_dim)
    // 位置pos的token，旋转角度 Φ[pos,j]=pos*θj    已经缓存下来了
    // c=cos(Φ[posj]),   s=sin(Φ[posj])
    // dst[2j]   =X[2j]*c - X[2j+1]*s
    // dst[2j+1] =X[2j]*s + X[2j+1]*c
    
    const float *w_base = static_cast<const float *>(src0->data);
    const float *pos_base = static_cast<const float *>(src1->data);
    const float *angle_table_base = static_cast<const float *>(src2->data);
    float *dst_base = static_cast<float *>(dst->data);

    // 先计算sin 和 cos
    const int64_t n_pairs=dims00/2;                     //有多少对  比如64对，每个token都共用64对angle
    std::vector<float> sin_cache(dims02 * n_pairs);     //每个token有n_pairs对，所以总的开的空间相乘 多头共用
    std::vector<float> cos_cache(dims02 * n_pairs);
    for (int64_t token = 0; token < dims02; ++token){

        const int64_t position = pos_base[token];       //获取当前token的实际位置

        const float* angle_table=angle_table_base[position*dim02];    //从angel表的哪里开始

        for (int64_t j = 0; j < n_pairs; ++j){

            const float angle = angle_table[j];

            const int64_t offset = token * n_pairs + j;

            sin_cache[offset] = std::sin(angle);
            cos_cache[offset] = std::cos(angle);
        }
    }

    _threadPool.parallel_for(dims01 * dims02 * dims03, [&](int64_t i){
        // 一个任务按照一个头去划分

        // 计算是第几个token
        const int64_t token = i / dims01;
        // 计算 从哪里开始取数  从哪里开始放
        const int64_t offset = i * dims00;              //元素的个数
        const int64_t trig_offset = token * n_pairs;
        
        const float *start = x_base + offset;
        float *dst = dst_base + offset;
        
        const float *sin_head = sin_cache.data() + trig_offset;
        const float *cos_head = cos_cache.data() + trig_offset;

        for (int64_t j = 0; j < n_pair ; ++j){
            // 已经缓存好的sin cos
            const float x0 = start[2 * j];
            const float x1 = start[2 * j + 1];
            const float c = cos_head[j];
            const float s = sin_head[j];
            y_head[2 * j] = x0 * c - x1 * s;
            y_head[2 * j + 1] = x0 * s + x1 * c;
        }
    });
    return {};
}

std::expected<void, std::error_code> CPUBackend::compute_flash_attn(Tensor* dst){
    // q[head_dim, n_heads,n_tokens_q]
    // k[head_dim, n_heads_kv,n_tokens_kv]
    // v[head_dim, n_heads_kv,n_tokens_kv]
    // O[head_dim, n_heads,n_tokens_q]

    
    //llama.cpp为了适配mul_mat接口采用permute 
    //这里TODO：暂时不引入permute  flashattention  kvcache(由计算图或缓存模块负责)

    Tensor *src0 = dst->src[0];             // q
    Tensor *src1 = dst->src[1];            // k
    Tensor *src2 = dst->src[2];            // v
    INFERLLM_TENSOR_TERNARY_LOCALS

    // 是不是GQA
    int64_t group_size=dims01/dims11;   //  Hq / Hkv (每多少个Q共用一个kv)

    const float *q_base = static_cast<const float *>(src0->data);
    const float *k_base = static_cast<const float *>(src1->data);
    const float *v_base = static_cast<const float *>(src2->data);
    float *dst_base = static_cast<float *>(dst->data);

    //softmax(qk/根号d)*v
    _threadPool.parallel_for(dims01*dims02,[&](int64_t i){
        // 这里我暂时实现的是S的一行作为一个任务 那么就是按照q的某个token的某个head作为一个任务

        // 先计算是第几个token的第几个头
        int64_t q_token = i / dims01;       //是第几个token
        int64_t q_head = i % dims01;        //是第几个head
        
        // 计算src的位置，计算dst的位置，相乘再相加
        const int64_t kv_head = q_head / group_size;                        // kv是用的哪个头，编号
        const int64_t q_offset = i * dims00;                                // q:当前头是跳过了多少个
        const float *q_start = q_base + q_offset;                           // q找到起始位置
        
        
        float *dst_start = dst_base + q_offset;                             // dst找到位置

        //对应位置相乘相加，然后放到指定位置即可
        
        // QK^T / sqrt(D)
        // 当前 Query 对所有 Key 的 Attention Score
        std::vector<float> scores(dims12);          //dims12=Nk
        float scale=std::sqrt(dims00);              // sqrt(D)
        for (int64_t key_token = 0; key_token < dims02; ++key_token){
            // mask : 只允许访问当前位置之前的token
            if (key_token > q_token){
                //当前 Query 正在和第几个 Key token 计算注意力分数
                scores[k] = -std::numeric_limits<float>::infinity();
                continue;
            }
            const int64_t kv_offset = (key_token * dims11 + kv_head) * dims00;      //计算偏移，因为我们算的是s的一行*整个kv矩阵，所以这里的kv_offset是每次循环会跳跃的，比如token0的head1，token1的head1 token2的head1

            const float *k_start = k_base + kv_offset;

            float dot = 0.0f;

            for (int64_t d = 0; d < dims00; ++d)
            {
                dot += q_start[d] * k_start[d];
            }

            scores[key_token] = dot * scale;
        }
        // softmax
        const float max_score = *std::max_element(scores.begin(), scores.end());        //找到最大值
        float sum=0.0f;
        for (float &val : scores){
            val = std::exp(val - max_score);
            sum += val;
        }
        for (float &val : scores){
            val /= sum;
        }
        // 和v进行相乘
        // std::vector<float> result(dims20);
        // for (int64_t d = 0; d < dims02; ++d){
        //     for (int64_t k = 0; k < dims20; k++){
        //         result[d] = v_start[k] * scores[d];
        //     }
        // }
        
        // 当前任务的输出是一个 D 维向量
        std::fill(dst_start, dst_start + dims00, 0.0f);

        // 遍历 V 的每一行，即不同 Key token 的 V 向量
        for (int64_t key_token = 0; key_token < dims12; ++key_token){

            const int64_t kv_offset = (key_token * dims21 + kv_head) * dims20;

            const float *v_start = v_base + kv_offset;

            const float weight = scores[key_token];

            // 当前 V 行向量乘以权重，累加到输出
            for (int64_t d = 0; d < dims20; ++d){
                dst_start[d] += weight * v_start[d];
            }
        }
    });
    return {};
}



std::expected<void, std::error_code> CPUBackend::compute_add(Tensor* dst){
    Tensor* src0=dst->src[0];
    Tensor* src1=dst->src[1];
    INFERLLM_TENSOR_BINARY_LOCALS
    
    //a[dims,token]
    const float* a_base=static_cast<const float*>(src0->data);
    const float* b_base=static_cast<const float*>(src1->data);
    float* dst_base=static_cast<float*>(dst->data);
    
    _threadPool.parallel_for(dims01,[&](int64_t i){
        //按照一行一个任务
        const float* a_start=a_base+i*dims00;
        const float* b_start=b_base+i*dims00;
        float* dst_start=dst_base+i*dims00;
        
        for(int64_t k=0;k<dims00;k++){
            dst_start[k]=a_start[k]+b_start[k];
        }
    });
    return {};
}


std::expected<void, std::error_code> CPUBackend::compute_silu(Tensor* dst){
    Tensor* src0=dst->src[0];
    INFERLLM_TENSOR_UNARY_LOCALS
    const float* src_base=static_cast<const float*>(src0->data);
    float* dst_base=static_cast<float*>(dst->data);

    _threadPool.parallel_for(dims01, [&](int64_t i){
        //一行一个任务      TODO:decode阶段只有一个token
        const float *src = src_base + i * dims00;
        float *dst = dst_base + i * dims00;
        for(int64_t k=0;k<dims00;k++){
            dst[k]= src[k] / (1.0f + std::exp(-src[k]));
        } 
    });
    return {};
}



std::expected<void, std::error_code> CPUBackend::compute_mul(Tensor* dst){
    Tensor *src0 = dst->src[0];
    Tensor *src1 = dst->src[1];

    INFERLLM_TENSOR_BINARY_LOCALS

    const float *a_base = static_cast<const float *>(src0->data);
    const float *b_base = static_cast<const float *>(src1->data);
    float *dst_base = static_cast<float *>(dst->data);

    _threadPool.parallel_for(dims01, [&](int64_t i){

        const float* a_start = a_base + i * dims00;
        const float* b_start = b_base + i * dims00;
        float* dst_start = dst_base + i * dims00;

        for (int64_t k = 0; k < dims00; ++k) {
            dst_start[k] = a_start[k] * b_start[k];
        } 
    });

    return {};
}
