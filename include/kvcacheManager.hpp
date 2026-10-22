#pragma once
class KVStrategy{
public:
    virtual ~KVStrategy() = default;
    virtual std::expected<void, std::error_code>
    init(const KVMeta &, const std::vector<Backend *> &layer_dev, const KVCacheConfig &) = 0;

    virtual std::expected<uint32_t, std::error_code> match_prefix(uint32_t seq, std::span<const int32_t> tokens) = 0;
    virtual std::expected<void, std::error_code> attach(uint32_t seq, uint32_t n_past, uint32_t n_new) = 0;
    virtual std::expected<KVLayerView, std::error_code> layer_view(uint32_t seq, int il) const = 0;
    virtual void commit(uint32_t seq, uint32_t n_tokens) = 0;
    virtual void release(uint32_t seq) = 0;

    virtual size_t reserved_bytes(const Backend *) const = 0;
    virtual bool owns(const Tensor *) const = 0;
};


class ContiguousKVStrategy : public KVStrategy { ... };
class PagedKVStrategy      : public KVStrategy { ... };
class RadixPagedKVStrategy : public PagedKVStrategy { ... };

struct KVMeta {                 // 从 HParams 构造，manager 初始化时给
    uint32_t n_layer = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_embd_head = 0;
    DataType k_type = GGML_TYPE_F16;
    DataType v_type = GGML_TYPE_F16;
};

struct KVCacheConfig {
    enum class Policy { Contiguous, Paged, RadixPaged } policy = Policy::Contiguous;
    uint32_t n_ctx = 0;         // 每序列最大长度
    uint32_t n_seq_max = 1;
    uint32_t block_size = 16;   // 仅 paged
};

// 建图唯一需要的东西：一层的 KV 在哪、多长、怎么写
struct KVLayerView {
    void*       k_base = nullptr;     // [0, capacity) 的 K 起点
    void*       v_base = nullptr;
    int64_t     k_row_bytes = 0;      // 每 token 占多少字节
    int64_t     v_row_bytes = 0;
    uint32_t    n_past = 0;
    uint32_t    n_kv = 0;             // n_past + n_tokens
    uint32_t    capacity = 0;
    Backend*    backend = nullptr;
    const int32_t* block_table = nullptr;  // paged 时非空
    uint32_t    n_blocks = 0;
};

// move-only，值语义
class KVContext{    
public:
    KVCacheManager *manager = nullptr;
    uint32_t seq_id = 0;
    uint32_t n_past = 0;
    uint32_t n_tokens = 0;
    uint32_t n_ctx() const { return n_past + n_tokens; }
};


class KVCacheManager{
public:
    // model.plan_weights_placement之后调用init
    std::expected<void, std::error_code> init(const KVMeta &meta, const std::vector<Backend *> &layer_dev, const KVCacheConfig &cfg);

    // 请求生命周期      查前缀 → n_past，容量检查
    std::expected<KVContext, std::error_code> begin_request(std::span<const int32_t> prompt, uint32_t n_new_tokens);
    void commit(KVContext &ctx, uint32_t n_tokens);
    void release(const KVContext &ctx);

    // 建图
    std::expected<KVLayerView, std::error_code> layer_view(const KVContext &ctx, int il) const;

    // Scheduler 用
    size_t reserved_bytes(const Backend *b) const;
    bool is_kv_tensor(const Tensor *t) const;

private:
    std::unique_ptr<KVStrategy> _strategy;                  //策略
    std::vector<Backend *> _layer_dev;                      
    KVMeta _meta;
    std::unordered_map<const Backend *, size_t> _reserved;
};

