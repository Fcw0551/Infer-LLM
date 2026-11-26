#pragma once
// 分页存储策略：每条序列的每一层按 block_tokens 个 token 划分逻辑页
// 页号 = token 绝对位置 / block_tokens，页内偏移 = token 绝对位置 % block_tokens
// 每页用一个 StorageId 引用该层所有 KV head 的 K/V 数据，物理空间由 KVStorage 分配

// --------------- 负责维护 token → StorageId 的映射状态 ----------------------

#include "../kvstrategy.hpp"

class BlockKVStrategy final : public KVStrategy {
public:
    using KVStrategy::KVStrategy;

    // 初始化底层存储，保存层信息和分页配置
    std::expected<void,std::error_code> init(std::span<const KVLayerSpec>,std::span<Backend* const>,const KVCacheConfig&) override;

    // 创建序列，为每层建立空页表
    std::expected<void,std::error_code> create_sequence(SequenceId) override;

    // 为本轮新增的位置预留空间，跨页时向底层申请缺少的页
    std::expected<void,std::error_code> reserve_append(SequenceId,TokenRange) override;

    // 确认本轮预留，将本轮刚写满的页交给底层持久化
    std::expected<void,std::error_code> commit_append(SequenceId,TokenRange) override;

    // 撤销本轮预留，释放新增页，保留追加前已有的尾页
    std::expected<void,std::error_code> abort_append(SequenceId,TokenRange) override;

    // 释放该序列各层的存储对象并移除页表
    std::expected<void,std::error_code> release_sequence(SequenceId) override;

    // 释放所有序列的存储对象并清空页表
    std::expected<void,std::error_code> clear_sequence() override;
private:
    
    // std::unique_ptr<KVStorage> _storage;
    // std::vector<KVLayerSpec> _layers;
    // KVCacheConfig _cfg = {};

    // 单条序列的页表，table[层号][逻辑页号] 对应一个 StorageId
    using Table = std::vector<std::vector<StorageId>>;
    std::unordered_map<SequenceId,Table> _sequences;

    // 计算容纳给定数量的 token 需要多少页，尾页计作一页 （也就是向上取整，找一下需要多少页）
    size_t pages(size_t tokens) const { return tokens ? (tokens-1)/_cfg.block_tokens+1 : 0; }

    // 每层保留前 pages 页，释放其余页对应的存储对象
    void truncate(Table&, size_t pages) noexcept;

    // 供基类 acquire_layer 使用，按页边界把 token 区间拆成带页内偏移的 K/V 片段
    std::expected<KVMapping,std::error_code> lookup(SequenceId,size_t,TokenRange) const override;

};
