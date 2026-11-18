#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "../kvutil.hpp"

// 上层将执行顺序解析成存储对象提示，不向策略暴露模型或计算图
// history_tokens 是已提交、需要上传的前缀长度；不包含尚未生成的尾部
struct KVPrefetchRequest {
    StorageId id = kInvalidStorageId;
    std::size_t history_tokens = 0;
};

struct KVPrefetchContext {
    StorageId current = kInvalidStorageId;
    std::span<const KVPrefetchRequest> upcoming;
};

class KVPrefetchPolicy {
public:
    virtual ~KVPrefetchPolicy() = default;

    // 返回按优先级排列的建议；Storage 负责去重、范围检查和预算准入
    // 第一版仅允许从 upcoming 选取；不持有 span，不执行搬运
    virtual std::vector<KVPrefetchRequest> plan(const KVPrefetchContext& context) const = 0;
};

class NoKVPrefetchPolicy final : public KVPrefetchPolicy {
public:
    std::string name(){
        return "None";
    }
    std::vector<KVPrefetchRequest> plan(const KVPrefetchContext& context) const override;
};

class NextAccessKVPrefetchPolicy final : public KVPrefetchPolicy {
public:
  std::string name(){
        return "NextAccess";
    }
    explicit NextAccessKVPrefetchPolicy(std::size_t lookahead);
    std::vector<KVPrefetchRequest> plan(const KVPrefetchContext& context) const override;
   
private:
    std::size_t _lookahead;
};
