#pragma once
#include "util.h"
#include <vector>
#include <span>
#include <expected>
#include <cstddef>
#include <string_view>
#include <system_error>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include<cstdint>
#include <cassert>
#include <fcntl.h>
#include <array>
#include<unordered_map>
#include <variant>
#include <string>
#include <cstring>
#include <type_traits>
//GGUF

//tensor元信息
struct TensorInfo {
    // 计算张量元素总数
    uint64_t num_elements() const noexcept {
        uint64_t n = 1;
        for (uint32_t i = 0; i < _n_dims; ++i) n *= _dims[i];
        return n;
    }
    // 获取有效维度视图（忽略未使用的维度）
    std::span<const uint64_t> shape() const noexcept {
        return {_dims.data(), _n_dims};
    }

    std::string _name;                              // 如 "blk.0.attn_q.weight"
    uint32_t    _n_dims;                            // 维度数，1~4
    std::array<uint64_t, GGUF_MAX_DIMS> _dims;       // 维度大小
    TensorType    _type;                               // 数据类型
    uint64_t    _offset;                             // 相对于数据区起始的字节偏移
};

//kv元信息
using MetaScalar = std::variant<
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
    float, double, bool, std::string
>;

struct MetaValue {
    GGUFType type;
    MetaScalar scalar;                      // type != ARRAY 有效
    GGUFType array_elem_type;               // type == ARRAY 有效：数组内元素是什么标量类型
    std::vector<MetaScalar> array_elems;    // type == ARRAY 有效：数组元素，无key
};

struct GGUFHeader {
    uint32_t magic;                                 // 必须是 0x46554747 = "GGUF" 小端序
    uint32_t version;                               // 当前 = 3
    uint64_t tensor_count;                          // 张量数量
    uint64_t metadata_kv_count;                     // 元数据键值对数量
};

//RAII管理GGUF文件
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    // 禁止拷贝，允许移动
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    // 工厂方法
    static std::expected<MappedFile, std::error_code> open(const std::string& path);

    // 只读视图（整个文件）
    std::span<const std::byte> view() const {
        return {_data, _size};
    }

    size_t size() const { return _size; }
    bool is_valid() const { return _data != nullptr; }

private:
   MappedFile(int fd, const std::byte* data, size_t size) noexcept
        : _fd(fd), _data(data), _size(size) 
    {}
    void reset();

    int _fd = -1;
    const std::byte* _data = nullptr;
    size_t _size = 0;
};
//kv元数据独立容器
class GGUFMetadata {
public:
    GGUFMetadata(){
        _metadata.clear();
    }
    ~GGUFMetadata(){
        _metadata.clear();
    }
    //插入
    void push(const std::string&k, const MetaValue&v){
        _metadata.insert({k,v});
    }
    // 查找，找到返回指针，没找到返回nullptr
    const MetaValue* find(std::string_view key) const{
        auto it = _metadata.find(std::string(key));   // unordered_map 默认不支持 string_view 异构查找
        if (it != _metadata.end()){
            return &(it->second);
        }
        return nullptr;
    }
    size_t size() const { return _metadata.size(); }
    void print() const;   
private:
    std::unordered_map<std::string,MetaValue> _metadata;
};
class GGUFContext{
public:
    GGUFContext() = default;
    GGUFContext(const GGUFContext&) = delete;
    GGUFContext& operator=(const GGUFContext&) = delete;
    
    GGUFContext(GGUFContext&&) noexcept = default;
    GGUFContext& operator=(GGUFContext&&) noexcept = default;

    static std::expected<GGUFContext, std::error_code> load(const std::string& path);

    const GGUFHeader& header() const { return _gguf_header; }
    const std::vector<TensorInfo>& tensors() const { return _tensors; }
    const GGUFMetadata& metadata() const { return _metadata; }

    //打印kv和tensor相关信息
    void print(size_t tensor_limit = 0) const;
private:
    std::expected<void, std::error_code> parser();
    std::expected<void, std::error_code> parser_header(std::span<const std::byte>&cur);
    std::expected<void, std::error_code> parser_metadata(std::span<const std::byte>&cur);
    std::expected<void, std::error_code> parser_tensorInfos(std::span<const std::byte>&cur);

    // KV 元数据读取辅助
    std::expected<MetaScalar, std::error_code> read_meta_scalar(std::span<const std::byte>& cur, GGUFType type);
    std::expected<MetaValue,  std::error_code> read_meta_value (std::span<const std::byte>& cur, GGUFType type);

    template<class T>
    std::expected<T, std::error_code> read_pod(std::span<const std::byte>& cur);
    std::expected<std::string, std::error_code> read_string(std::span<const std::byte>& cur);

    //打印相关
    void print_header() const;
    void print_metadata() const;
    void print_tensors(size_t limit = 0) const;

private:
    GGUFHeader _gguf_header;
    GGUFMetadata _metadata;
    std::vector<TensorInfo> _tensors;
    MappedFile _file;
    uint32_t _alignment = 32;           // general.alignment，数据区对齐，默认 32
    size_t   _data_region_offset = 0;   // 数据区起始（相对文件头）
};
