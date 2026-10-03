// inferllm/log.h
// 日志系统：多级别、线程安全、printf 格式化、终端着色、自定义回调
// 设计参考 llama.cpp 的全局日志回调 (llama_log_set / ggml_log_set)
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <functional>
#include <mutex>
#include <source_location>
#include <vector>
#include <optional>
#include <thread>

namespace inferllm {

// 日志级别：数值越大级别越高，低于全局级别的会被过滤
enum class LogLevel : int {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    Fatal = 5,
    Count = 6,
};

const char* log_level_name(LogLevel level) noexcept;
std::optional<LogLevel> log_level_from_string(std::string_view name) noexcept;

// 传递给回调的完整日志消息
struct LogMessage {
    LogLevel       level;
    std::string    text;
    std::string    file;        // 不含路径的文件名
    int            line;
    std::string    function;
    int64_t        timestamp_ms; // 进程启动后毫秒数
    std::thread::id thread_id;
};

using LogCallback = std::function<void(const LogMessage&)>;

class Logger {
public:
    Logger();

    void set_level(LogLevel level) noexcept;
    LogLevel level() const noexcept;

    void set_color_enabled(bool enabled) noexcept;
    bool color_enabled() const noexcept;

    void set_timestamp_enabled(bool enabled) noexcept;
    bool timestamp_enabled() const noexcept;

    void set_location_enabled(bool enabled) noexcept;
    bool location_enabled() const noexcept;

    // 设置主回调，设置后不再输出到 stderr，传 nullptr 恢复默认
    void set_callback(LogCallback callback) noexcept;

    // 添加额外回调（与默认输出并存，可用于同时写文件）
    size_t add_callback(LogCallback callback);
    void remove_callback(size_t id);

    // 直接输出字符串
    void log(LogLevel level,
             std::string_view text,
             std::source_location loc = std::source_location::current());

    // printf 风格格式化，GCC/Clang 会做编译期格式串检查
    void logf(LogLevel level,
              std::source_location loc,
              const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
        __attribute__((format(printf, 4, 5)))
#endif
        ;

    bool should_log(LogLevel level) const noexcept;

private:
    void emit(const LogMessage& msg);
    void emit_default(const LogMessage& msg);

    mutable std::mutex mutex_;
    LogLevel  level_             = LogLevel::Info;
    bool      color_enabled_     = true;
    bool      timestamp_enabled_ = true;
    bool      location_enabled_  = true;
    LogCallback primary_callback_;
    std::vector<std::pair<size_t, LogCallback>> extra_callbacks_;
    size_t  next_callback_id_ = 1;
    int64_t start_time_ms_    = 0;
};

namespace log {
Logger& logger() noexcept;
inline void set_level(LogLevel level)    { logger().set_level(level); }
inline void set_color(bool enabled)      { logger().set_color_enabled(enabled); }
inline void set_callback(LogCallback cb) { logger().set_callback(std::move(cb)); }
inline bool should_log(LogLevel level)   { return logger().should_log(level); }
} // namespace log

// 用户主要使用这些宏
#define LOG_TRACE(...) inferllm::log::logger().logf(inferllm::LogLevel::Trace, std::source_location::current(), __VA_ARGS__)
#define LOG_DEBUG(...) inferllm::log::logger().logf(inferllm::LogLevel::Debug, std::source_location::current(), __VA_ARGS__)
#define LOG_INFO(...)  inferllm::log::logger().logf(inferllm::LogLevel::Info,  std::source_location::current(), __VA_ARGS__)
#define LOG_WARN(...)  inferllm::log::logger().logf(inferllm::LogLevel::Warn,  std::source_location::current(), __VA_ARGS__)
#define LOG_ERROR(...) inferllm::log::logger().logf(inferllm::LogLevel::Error, std::source_location::current(), __VA_ARGS__)
#define LOG_FATAL(...) inferllm::log::logger().logf(inferllm::LogLevel::Fatal, std::source_location::current(), __VA_ARGS__)

// 条件日志
#define LOG_TRACE_IF(cond, ...) do { if (cond) LOG_TRACE(__VA_ARGS__); } while(0)
#define LOG_DEBUG_IF(cond, ...) do { if (cond) LOG_DEBUG(__VA_ARGS__); } while(0)
#define LOG_INFO_IF(cond, ...)  do { if (cond) LOG_INFO(__VA_ARGS__);  } while(0)
#define LOG_WARN_IF(cond, ...)  do { if (cond) LOG_WARN(__VA_ARGS__);  } while(0)
#define LOG_ERROR_IF(cond, ...) do { if (cond) LOG_ERROR(__VA_ARGS__); } while(0)

// Release 模式下零开销的调试日志
#ifdef NDEBUG
#define LOG_DTRACE(...) ((void)0)
#define LOG_DDEBUG(...) ((void)0)
#define LOG_DINFO(...)  ((void)0)
#else
#define LOG_DTRACE(...) LOG_TRACE(__VA_ARGS__)
#define LOG_DDEBUG(...) LOG_DEBUG(__VA_ARGS__)
#define LOG_DINFO(...)  LOG_INFO(__VA_ARGS__)
#endif

} // namespace inferllm
