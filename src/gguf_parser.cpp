#include "../include/gguf_parser.hpp"
#include <cstdio>
std::expected<MappedFile, std::error_code> MappedFile::open(const std::string& path){
    //open file
    int fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC);
    if(fd==-1){
        return std::unexpected(std::error_code(errno, std::generic_category()));
    }
    //file元信息
    struct stat st;
    if(::fstat(fd, &st) == -1){
        auto err = std::error_code(errno, std::generic_category());
        ::close(fd); // 失败必须关闭 fd，否则泄漏
        return std::unexpected(err);
    }
    //检查file的类型
    if(!S_ISREG(st.st_mode)) {
        ::close(fd);
        return std::unexpected(std::error_code(EINVAL, std::generic_category()));
    }
    //处理空文件
    if(st.st_size == 0) {
        ::close(fd);
        return MappedFile(-1, nullptr, 0);  // 有效但空的对象
    }
    //mmap 
    void*raw=::mmap(nullptr,static_cast<size_t>(st.st_size),PROT_READ,MAP_PRIVATE,fd,0);
    if(raw==MAP_FAILED){
        auto err=std::error_code(errno,std::generic_category());
        ::close(fd);
        return std::unexpected(err);
    }
    //------------success--------------
    //优化建议
    ::madvise(raw, static_cast<size_t>(st.st_size), MADV_SEQUENTIAL);
    return MappedFile(fd,static_cast<const std::byte*>(raw),static_cast<size_t>(st.st_size));
}
void MappedFile::reset(){
    // 先 munmap（如果有映射）
    if(_data != nullptr) {
        // munmap 失败通常意味着参数错误，但在析构中无法报告，忽略
        ::munmap(const_cast<void*>(static_cast<const void*>(_data)), _size);
        _data = nullptr;
    }
    // 再 close（如果有 fd）
    if (_fd != -1) {
        ::close(_fd);
        _fd = -1;
    }
    _size = 0;
}

MappedFile:: ~MappedFile(){
    reset();
}
MappedFile::MappedFile(MappedFile&& other) noexcept{
    assert(&other!=this);
    _fd=other._fd;
    _data=other._data;
    _size=other._size;
    other._fd=-1;
    other._data=nullptr;
    other._size=0;
}
MappedFile& MappedFile::operator=(MappedFile &&other) noexcept{
    assert(&other != this);
    reset();
    _fd = other._fd;
    _data = other._data;
    _size = other._size;
    other._fd = -1;
    other._data = nullptr;
    other._size = 0;
    return *this;
}

template<class T>
std::expected<T, std::error_code> GGUFContext::read_pod(std::span<const std::byte>& cur) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (cur.size() < sizeof(T)) {
        return std::unexpected(std::make_error_code(std::errc::message_size));
    }
    T v;
    std::memcpy(&v, cur.data(), sizeof(T));
    cur = cur.subspan(sizeof(T));
    return v;
}

// GGUF 字符串：uint64 长度 + 字节数据（无 null 终止）
std::expected<std::string, std::error_code> GGUFContext::read_string(std::span<const std::byte>& cur) {
    auto len = read_pod<uint64_t>(cur);
    if (!len) return std::unexpected(len.error());
    if (cur.size() < *len) {
        return std::unexpected(std::make_error_code(std::errc::message_size));
    }
    std::string s(reinterpret_cast<const char*>(cur.data()), *len);
    cur = cur.subspan(*len);
    return s;
}
std::expected<GGUFContext, std::error_code> GGUFContext::load(const std::string& path) {
    auto mf = MappedFile::open(path);
    if (!mf) return std::unexpected(mf.error());

    GGUFContext ctx;
    ctx._file = std::move(*mf);
    if (auto r = ctx.parser(); !r) return std::unexpected(r.error());
    return ctx;
}

std::expected<void, std::error_code> GGUFContext:: parser(){
    auto cursor = _file.view();

    if (auto r = parser_header(cursor); !r)
        return r;
    if (auto r = parser_metadata(cursor); !r)
        return r;
    if (auto r = parser_tensorInfos(cursor); !r)
        return r;

    // 计算数据区偏移（对齐）
    size_t consumed = cursor.data() - _file.view().data();
    size_t aligned = (consumed + _alignment - 1) / _alignment * _alignment;
    _data_region_offset = aligned;
    return {};
}
std::expected<void, std::error_code>  GGUFContext::parser_header( std::span<const std::byte>&cur){
    auto magic = read_pod<uint32_t>(cur);
    if (!magic) return std::unexpected(magic.error());
    if (*magic != 0x46554747u) {   // "GGUF" 小端
        return std::unexpected(std::make_error_code(std::errc::illegal_byte_sequence));
    }
    _gguf_header.magic = *magic;

    auto version = read_pod<uint32_t>(cur);
    if (!version) return std::unexpected(version.error());
    if (*version != 3) {
        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }
    _gguf_header.version = *version;

    auto n_tensors = read_pod<uint64_t>(cur);
    if (!n_tensors) return std::unexpected(n_tensors.error());
    _gguf_header.tensor_count = *n_tensors;

    auto n_kv = read_pod<uint64_t>(cur);
    if (!n_kv) return std::unexpected(n_kv.error());
    _gguf_header.metadata_kv_count = *n_kv;

    return {};
}
// 读取单个标量 KV 值
std::expected<MetaScalar, std::error_code>
GGUFContext::read_meta_scalar(std::span<const std::byte>& cur, GGUFType type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   { auto v = read_pod<uint8_t>(cur);  if (!v) return std::unexpected(v.error()); return MetaScalar{*v}; }
        case GGUF_TYPE_INT8:    { auto v = read_pod<int8_t>(cur);   if (!v) return std::unexpected(v.error()); return MetaScalar{*v}; }
        case GGUF_TYPE_UINT16:  { auto v = read_pod<uint16_t>(cur); if (!v) return std::unexpected(v.error()); return MetaScalar{*v}; }
        case GGUF_TYPE_INT16:   { auto v = read_pod<int16_t>(cur);  if (!v) return std::unexpected(v.error()); return MetaScalar{*v}; }
        case GGUF_TYPE_UINT32:  { auto v = read_pod<uint32_t>(cur); if (!v) return std::unexpected(v.error()); return MetaScalar{*v}; }
        case GGUF_TYPE_INT32:   { auto v = read_pod<int32_t>(cur);  if (!v) return std::unexpected(v.error()); return MetaScalar{*v}; }
        case GGUF_TYPE_UINT64:  { auto v = read_pod<uint64_t>(cur); if (!v) return std::unexpected(v.error()); return MetaScalar{*v}; }
        case GGUF_TYPE_INT64:   { auto v = read_pod<int64_t>(cur);  if (!v) return std::unexpected(v.error()); return MetaScalar{*v}; }
        case GGUF_TYPE_FLOAT32: { auto v = read_pod<float>(cur);    if (!v) return std::unexpected(v.error()); return MetaScalar{*v}; }
        case GGUF_TYPE_FLOAT64: { auto v = read_pod<double>(cur);   if (!v) return std::unexpected(v.error()); return MetaScalar{*v}; }
        // GGUF 的 bool 占 1 字节
        case GGUF_TYPE_BOOL:    { auto v = read_pod<uint8_t>(cur);  if (!v) return std::unexpected(v.error()); return MetaScalar{*v != 0}; }
        case GGUF_TYPE_STRING:  { auto v = read_string(cur);        if (!v) return std::unexpected(v.error()); return MetaScalar{std::move(*v)}; }
        default:  // GGUF_TYPE_ARRAY 在此非法；越界类型
            return std::unexpected(std::make_error_code(std::errc::illegal_byte_sequence));
    }
}

// 读取一个 KV 值：标量或数组
std::expected<MetaValue, std::error_code>
GGUFContext::read_meta_value(std::span<const std::byte>& cur, GGUFType type) {
    MetaValue mv;
    mv.type = type;

    if (type != GGUF_TYPE_ARRAY) {
        auto s = read_meta_scalar(cur, type);
        if (!s) return std::unexpected(s.error());
        mv.scalar = std::move(*s);
        return mv;
    }

    // 数组：u32 元素类型 + u64 元素个数 + 元素序列（元素无各自的类型标签/key）
    auto et = read_pod<uint32_t>(cur);
    if (!et) return std::unexpected(et.error());
    GGUFType elem = static_cast<GGUFType>(*et);
    if (elem == GGUF_TYPE_ARRAY || elem >= GGUF_TYPE_COUNT)  // 禁止嵌套数组
        return std::unexpected(std::make_error_code(std::errc::illegal_byte_sequence));
    mv.array_elem_type = elem;

    auto n = read_pod<uint64_t>(cur);
    if (!n) return std::unexpected(n.error());

    // 定长元素的数组先按剩余字节做上限校验，避免损坏文件导致巨量 reserve/OOM
    size_t elem_size = 0;
    switch (elem) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: elem_size = 1; break;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: elem_size = 2; break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: elem_size = 4; break;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: elem_size = 8; break;
        default: elem_size = 0; break;  // STRING：变长，不做上限估计
    }
    if (elem_size != 0 && *n > cur.size() / elem_size)
        return std::unexpected(std::make_error_code(std::errc::message_size));

    mv.array_elems.reserve(static_cast<size_t>(*n));
    for (uint64_t i = 0; i < *n; ++i) {
        auto e = read_meta_scalar(cur, elem);
        if (!e) return std::unexpected(e.error());
        mv.array_elems.push_back(std::move(*e));
    }
    return mv;
}

std::expected<void, std::error_code> GGUFContext:: parser_metadata(std::span<const std::byte>&cur){
    for (uint64_t i = 0; i < _gguf_header.metadata_kv_count; ++i) {
        auto key = read_string(cur);
        if (!key) return std::unexpected(key.error());

        auto t = read_pod<uint32_t>(cur);
        if (!t) return std::unexpected(t.error());
        GGUFType type = static_cast<GGUFType>(*t);
        if (type >= GGUF_TYPE_COUNT)
            return std::unexpected(std::make_error_code(std::errc::illegal_byte_sequence));

        auto val = read_meta_value(cur, type);
        if (!val) return std::unexpected(val.error());

        // general.alignment 决定数据区对齐（缺省 32）
        if (*key == "general.alignment" && val->type == GGUF_TYPE_UINT32) {
            _alignment = std::get<uint32_t>(val->scalar);
        }

        _metadata.push(*key, *val);
    }
    return {};
}
std::expected<void, std::error_code>  GGUFContext::parser_tensorInfos(std::span<const std::byte>&cur){
    _tensors.reserve(_gguf_header.tensor_count);   
    for (uint64_t i = 0; i < _gguf_header.tensor_count; ++i){
        //[uint64_t][name][uint32][uint64][uint32][uint64]
        TensorInfo info;

        // name GGUF string
        auto name = read_string(cur);
        if (!name)return std::unexpected(name.error());
        info._name = std::move(*name);

        // n_dimensions
        auto n_dims = read_pod<uint32_t>(cur);
        if (!n_dims)return std::unexpected(n_dims.error());
        if (*n_dims == 0 || *n_dims > GGUF_MAX_DIMS)return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        info._n_dims = *n_dims;

        //dimensions 数组
        for (uint32_t d = 0; d < *n_dims; ++d)
        {
            auto dim = read_pod<uint64_t>(cur);
            if (!dim)
                return std::unexpected(dim.error());
            info._dims[d] = *dim;
        }

        //type
        auto type = read_pod<uint32_t>(cur);
        if (!type)
            return std::unexpected(type.error());
        info._type = static_cast<DataType>(*type);

        //offset
        auto offset = read_pod<uint64_t>(cur);
        if (!offset)
            return std::unexpected(offset.error());
        info._offset = *offset;

        _tensors.push_back(std::move(info));
        _tensor_index[_tensors.back()._name]=&_tensors.back();   //info 已被 move 走，名字要从向量里取
    }
    //整个弄完之后，要进行一个data_offset
    size_t consumed = cur.data() - _file.view().data();
    _data_region_offset = (consumed + _alignment - 1) / _alignment * _alignment;
    return {};
}





//-----------------打印相关------------------

// 类型名统一用 util.h 里的 gguf_value_type_name / ggml_type_name（inline）

// 打印一个 MetaScalar（variant）
static void print_scalar(const MetaScalar& v) {
    std::visit([](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::string>) {
            std::printf("\"%s\"", x.c_str());
        } else if constexpr (std::is_same_v<T, bool>) {
            std::printf("%s", x ? "true" : "false");
        } else if constexpr (std::is_floating_point_v<T>) {
            std::printf("%g", static_cast<double>(x));
        } else if constexpr (std::is_signed_v<T>) {
            std::printf("%lld", static_cast<long long>(x));
        } else {
            std::printf("%llu", static_cast<unsigned long long>(x));
        }
    }, v);
}

// 打印一个 KV 值（标量直接打；数组太长只展开前几个）
static void print_meta_value(const MetaValue& mv) {
    if (mv.type != GGUF_TYPE_ARRAY) {
        print_scalar(mv.scalar);
        return;
    }
    constexpr size_t kMaxShow = 8;
    const size_t total = mv.array_elems.size();
    const size_t show  = total < kMaxShow ? total : kMaxShow;

    std::printf("ARRAY<%s> n=%zu [", gguf_value_type_name(mv.array_elem_type), total);
    for (size_t i = 0; i < show; ++i) {
        if (i) std::printf(", ");
        print_scalar(mv.array_elems[i]);
    }
    if (total > kMaxShow) std::printf(", ... +%zu", total - kMaxShow);
    std::printf("]");
}

// 打印所有 KV 元数据
void GGUFMetadata::print() const {
    std::printf("========== GGUF Metadata (%zu entries) ==========\n", _metadata.size());
    for (const auto& [key, val] : _metadata) {
        std::printf("%-40s %-8s: ", key.c_str(), gguf_value_type_name(val.type));
        print_meta_value(val);
        std::printf("\n");
    }
    std::printf("\n");
}

//打印gguf的header
void GGUFContext::print_header() const {
    std::printf("========== GGUF Header ==========\n");
    std::printf("magic              : 0x%08X", _gguf_header.magic);
    if (_gguf_header.magic == 0x46554747u) std::printf(" (\"GGUF\")");
    std::printf("\n");
    std::printf("version            : %u\n", _gguf_header.version);
    std::printf("tensor_count       : %llu\n",
                static_cast<unsigned long long>(_gguf_header.tensor_count));
    std::printf("metadata_kv_count  : %llu\n",
                static_cast<unsigned long long>(_gguf_header.metadata_kv_count));
    std::printf("alignment          : %llu\n",
                static_cast<unsigned long long>(_alignment));
    std::printf("data_region_offset : %zu (0x%zX)\n",
                _data_region_offset, _data_region_offset);
    std::printf("file_size          : %zu\n", _file.size());
    std::printf("\n");
}

void GGUFContext::print_metadata() const {
    _metadata.print();
}

void GGUFContext::print_tensors(size_t limit) const {
    std::printf("========== GGUF Tensors (%zu total) ==========\n", _tensors.size());
    std::printf("%-4s %-40s %-8s %-8s %-12s %s\n",
                "#", "name", "type", "n_dims", "offset", "shape");

    size_t n = (limit == 0 || limit > _tensors.size()) ? _tensors.size() : limit;

    for (size_t i = 0; i < n; ++i) {
        const TensorInfo& t = _tensors[i];

        // shape 拼成 [d0, d1, d2, ...]
        std::string shape = "[";
        for (uint32_t d = 0; d < t._n_dims; ++d) {
            if (d > 0) shape += ", ";
            shape += std::to_string(t._dims[d]);
        }
        shape += "]";

        std::printf("%-4zu %-40s %-8s %-8u %-12llu %s\n",
                    i,
                    t._name.c_str(),
                    ggml_type_name(t._type),
                    t._n_dims,
                    static_cast<unsigned long long>(t._offset),
                    shape.c_str());
    }

    if (n < _tensors.size()) {
        std::printf("... (%zu more tensors omitted)\n", _tensors.size() - n);
    }
    std::printf("\n");
}

void GGUFContext::print(size_t tensor_limit) const {
    print_header();
    print_metadata();
    print_tensors(tensor_limit);
}