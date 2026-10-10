#include "backend.hpp"

//-------------BackendRegistry-----------------
    // 单例
    BackendRegistry& BackendRegistry::instance(){
        static BackendRegistry reg;
        return reg;
    }

    // 注册
    void BackendRegistry::register_backend(std::string type_name, ProbeFunc probe, CreateFunc creator){
       _entries[std::move(type_name)] = { std::move(probe), std::move(creator) };
    }

    // 探测
    void BackendRegistry::probe_all(){
        _devices.clear();

        for (auto &[type_name, entry] : _entries){
            auto infos = entry.probe();
            for (auto &info : infos){
                _devices.push_back(std::move(info));
            }
        }
    }
    // 创建实例
    std::expected<std::unique_ptr<Backend>, std::error_code> BackendRegistry::create(std::string_view device_name) const{
        // 先确认设备真实存在
        auto dev_it = std::find_if(_devices.begin(), _devices.end(), [&](const DeviceInfo &d) { return d.name == device_name; });

        if (dev_it == _devices.end()){
            return std::unexpected(std::make_error_code(std::errc::no_such_device));
        }

        // 解析类型和 device_id
        std::string type = extract_type(device_name);
        int device_id = extract_device_id(device_name);

        // 找注册的 create
        auto entry_it = _entries.find(type);
        if (entry_it == _entries.end()){
            return std::unexpected(std::make_error_code(std::errc::no_such_device));
        }
        return entry_it->second.create(device_id);
    }
    
    // 自动选择（优先 CUDA > Metal > CPU，可被环境变量覆盖）
    std::expected<std::unique_ptr<Backend>, std::error_code> BackendRegistry::create_default() const{
        // 环境变量
        if (const char* env = std::getenv("INFERLLM_DEVICE")) {
            return create(env);   
        }
        // 自动选择
        if (_devices.empty()){
            return std::unexpected(std::make_error_code(std::errc::no_such_device));
        }

        // CUDA > Metal > CPU
        if (_devices.empty()){
            return std::unexpected(std::make_error_code(std::errc::no_such_device));
        }

        for (auto type : {DeviceType::CUDA, DeviceType::METAL, DeviceType::CPU}){
            for (const auto &d : _devices){
                if (d.type == type)
                    return create(d.name);
            }
        }
        return std::unexpected(std::make_error_code(std::errc::no_such_device));
    }

    // 查询探测结果 (辅助函数)
    const std::vector<DeviceInfo>& BackendRegistry::available_devices() const{
        return _devices;
    }
    std::expected<DeviceInfo,std::error_code> BackendRegistry::device_info(std::string_view name) const{
        for(auto &e:_devices){
            if(e.name==name){
                return e;
            }
        }
        return std::unexpected(make_error_code(std::errc::invalid_argument));
    }

    // 解析辅助函数
    std::string BackendRegistry::extract_type(std::string_view name){
        auto colon = name.find(':');
        return colon == std::string_view::npos
                   ? std::string(name)
                   : std::string(name.substr(0, colon));
    }

    int BackendRegistry::extract_device_id(std::string_view name)
    {
        auto colon = name.find(':');
        if (colon == std::string_view::npos)
            return 0; // "cpu" → 0
        return std::stoi(std::string(name.substr(colon + 1)));
    }

//---------------- 统一注册-----------------
void register_all_backends() {
    register_cpu_backend();

#ifdef INFERLLM_WITH_CUDA
    register_cuda_backend();
#endif
}


