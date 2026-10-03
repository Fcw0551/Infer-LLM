// inferllm/log.cpp
#include "inferllm/log.h"

#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <cctype>

namespace inferllm {

// ============================================================================
// 工具函数
// ============================================================================

const char* log_level_name(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERROR";
        case LogLevel::Fatal: return "FATAL";
        default:              return "?????";
    }
}

std::optional<LogLevel> log_level_from_string(std::string_view name) noexcept {
    // 转小写比较
    std::string lower;
    lower.reserve(name.size());
    for (char c : name) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    if (lower == "trace") return LogLevel::Trace;
    if (lower == "debug") return LogLevel::Debug;
    if (lower == "info")  return LogLevel::Info;
    if (lower == "warn" || lower == "warning") return LogLevel::Warn;
    if (lower == "error") return LogLevel::Error;
    if (lower == "fatal") return LogLevel::Fatal;
    return std::nullopt;
}

// 从路径中提取文件名（不含目录）
static std::string_view basename(std::string_view path) noexcept {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string_view::npos) return path;
    return path.substr(pos + 1);
}

// 获取进程启动后经过的毫秒数（单调时钟，不受系统时间调整影响）
static int64_t elapsed_ms(int64_t start) noexcept {
    auto now = std::chrono::steady_clock::now();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return ms - start;
}

// 终端颜色码（ANSI escape sequences）
namespace ansi {
    static constexpr const char* RESET   = "\033[0m";
    static constexpr const char* GRAY    = "\033[90m";
    static constexpr const char* CYAN    = "\033[36m";
    static constexpr const char* GREEN   = "\033[32m";
    static constexpr const char* YELLOW  = "\033[33m";
    static constexpr const char* RED     = "\033[31m";
    static constexpr const char* MAGENTA = "\033[35m";
    static constexpr const char* BOLD    = "\033[1m";
}

static const char* level_color(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::Trace: return ansi::GRAY;
        case LogLevel::Debug: return ansi::CYAN;
        case LogLevel::Info:  return ansi::GREEN;
        case LogLevel::Warn:  return ansi::YELLOW;
        case LogLevel::Error: return ansi::RED;
        case LogLevel::Fatal: return ansi::MAGENTA;
        default:              return ansi::RESET;
    }
}

// 检测 stderr 是否连接到终端（TTY），非 TTY 时禁用颜色
static bool is_stderr_tty() noexcept {
#if defined(_WIN32)
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(fileno(stderr)) != 0;
#endif
}

// ============================================================================
// Logger 实现
// ============================================================================

Logger::Logger() {
    // 记录进程启动时间
    auto now = std::chrono::steady_clock::now();
    start_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now.time_since_epoch()).count();
    // 非终端环境默认关闭颜色
    color_enabled_ = is_stderr_tty();
}

void Logger::set_level(LogLevel level) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    level_ = level;
}

LogLevel Logger::level() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return level_;
}

void Logger::set_color_enabled(bool enabled) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    color_enabled_ = enabled;
}

bool Logger::color_enabled() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return color_enabled_;
}

void Logger::set_timestamp_enabled(bool enabled) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    timestamp_enabled_ = enabled;
}

bool Logger::timestamp_enabled() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return timestamp_enabled_;
}

void Logger::set_location_enabled(bool enabled) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    location_enabled_ = enabled;
}

bool Logger::location_enabled() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return location_enabled_;
}

void Logger::set_callback(LogCallback callback) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    primary_callback_ = std::move(callback);
}

size_t Logger::add_callback(LogCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t id = next_callback_id_++;
    extra_callbacks_.emplace_back(id, std::move(callback));
    return id;
}

void Logger::remove_callback(size_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    extra_callbacks_.erase(
        std::remove_if(extra_callbacks_.begin(), extra_callbacks_.end(),
                       [id](const auto& p) { return p.first == id; }),
        extra_callbacks_.end());
}

bool Logger::should_log(LogLevel level) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return level >= level_;
}

void Logger::log(LogLevel level, std::string_view text, std::source_location loc) {
    if (!should_log(level)) return;

    LogMessage msg;
    msg.level        = level;
    msg.text         = std::string(text);
    msg.file         = std::string(basename(loc.file_name()));
    msg.line         = static_cast<int>(loc.line());
    msg.function     = loc.function_name();
    msg.timestamp_ms = elapsed_ms(start_time_ms_);
    msg.thread_id    = std::this_thread::get_id();

    emit(msg);
}

void Logger::logf(LogLevel level, std::source_location loc, const char* fmt, ...) {
    if (!should_log(level)) return;

    // 第一次 vsnprintf 计算所需长度
    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);
    int needed = vsnprintf(nullptr, 0, fmt, args_copy);
    va_end(args_copy);

    std::string text;
    if (needed < 0) {
        text = fmt;  // 格式化失败，退化为原始格式串
    } else {
        text.resize(static_cast<size_t>(needed));
        vsnprintf(text.data(), static_cast<size_t>(needed) + 1, fmt, args);
    }
    va_end(args);

    LogMessage msg;
    msg.level        = level;
    msg.text         = std::move(text);
    msg.file         = std::string(basename(loc.file_name()));
    msg.line         = static_cast<int>(loc.line());
    msg.function     = loc.function_name();
    msg.timestamp_ms = elapsed_ms(start_time_ms_);
    msg.thread_id    = std::this_thread::get_id();

    emit(msg);
}

void Logger::emit(const LogMessage& msg) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 主回调：如果设置了，替代默认输出
    if (primary_callback_) {
        primary_callback_(msg);
    } else {
        emit_default(msg);
    }

    // 额外回调：始终调用（用于写文件等）
    for (auto& [id, cb] : extra_callbacks_) {
        if (cb) cb(msg);
    }
}

void Logger::emit_default(const LogMessage& msg) {
    // 格式: [时间] [级别] [文件:行] 消息
    // 示例: [00:01:23.456] [INFO ] [gguf_parser.cpp:42] magic=0x46554747

    char time_buf[32];
    if (timestamp_enabled_) {
        int64_t total_ms = msg.timestamp_ms;
        int hours   = static_cast<int>(total_ms / 3600000);
        int minutes = static_cast<int>((total_ms % 3600000) / 60000);
        int seconds = static_cast<int>((total_ms % 60000) / 1000);
        int millis  = static_cast<int>(total_ms % 1000);
        snprintf(time_buf, sizeof(time_buf), "%02d:%02d:%02d.%03d",
                 hours, minutes, seconds, millis);
    }

    const char* color = color_enabled_ ? level_color(msg.level) : "";
    const char* reset = color_enabled_ ? ansi::RESET : "";
    const char* bold  = (msg.level == LogLevel::Error || msg.level == LogLevel::Fatal)
                        && color_enabled_ ? ansi::BOLD : "";

    // 级别标签固定 5 字符宽度，对齐输出
    char level_buf[16];
    snprintf(level_buf, sizeof(level_buf), "%-5s", log_level_name(msg.level));

    if (timestamp_enabled_ && location_enabled_) {
        fprintf(stderr, "%s[%s] [%s%s%s] [%s:%d] %s%s\n",
                bold ? bold : "",
                time_buf,
                color, level_buf, reset,
                msg.file.c_str(), msg.line,
                msg.text.c_str(),
                bold ? reset : "");
    } else if (timestamp_enabled_) {
        fprintf(stderr, "%s[%s] [%s%s%s] %s%s\n",
                bold ? bold : "",
                time_buf,
                color, level_buf, reset,
                msg.text.c_str(),
                bold ? reset : "");
    } else if (location_enabled_) {
        fprintf(stderr, "%s[%s%s%s] [%s:%d] %s%s\n",
                bold ? bold : "",
                color, level_buf, reset,
                msg.file.c_str(), msg.line,
                msg.text.c_str(),
                bold ? reset : "");
    } else {
        fprintf(stderr, "%s[%s%s%s] %s%s\n",
                bold ? bold : "",
                color, level_buf, reset,
                msg.text.c_str(),
                bold ? reset : "");
    }

    fflush(stderr);
}

// ============================================================================
// 全局单例
// ============================================================================

namespace log {

Logger& logger() noexcept {
    // Meyers' Singleton，C++11 起保证线程安全初始化
    static Logger instance;
    return instance;
}

} // namespace log

} // namespace inferllm
