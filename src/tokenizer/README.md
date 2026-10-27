# Tokenizer

独立 C++23 模块，当前支持 GGUF `tokenizer.ggml.model=gpt2`、
`tokenizer.ggml.pre=qwen2`，用于当前 Qwen3 模型。未知策略返回错误。
不依赖 Model、Tensor、Backend、Scheduler 或 Sampler。

## 文件职责

- `include/tokenizer.hpp`：唯一公共接口；仅前置声明 `GGUFMetadata`。
- `tokenizer.cpp`：公共接口、策略分发、新增 token 匹配、错误消息。
- `tokenizer_data.hpp`：私有不可变词表、合并排名和匹配索引。
- `tokenizer_gguf.cpp`：元数据读取、严格校验、构建索引。
- `tokenizer_pre.cpp`：ICU NFC 规范化、PCRE2 Unicode 预分词。
- `tokenizer_bpe.cpp`：按排名、左侧优先执行 BPE；优先队列避免全量重复扫描。

## 使用

```cpp
#include "tokenizer.hpp"
#include "gguf_parser.hpp"

auto tokenizer = Tokenizer::from_gguf(gguf.metadata());
if (!tokenizer) { /* tokenizer.error().message() */ }
else {
    auto ids = tokenizer->encode("介绍一下你自己");
    if (ids) {
        auto text = tokenizer->decode(*ids);
    }
}
```

Tokenizer 拥有加载的数据，GGUF 释放后仍可使用。复制 Tokenizer 共享只读词表，
并发 encode/decode 使用各自的临时工作区。

- 输入必须是有效 UTF-8，可包含 NUL。Qwen 普通文本按 NFC 规范化，因此
  `decode(encode(text))` 可能返回规范化后的文本，而非原始字节。
- `EncodeOptions::parse_special` 默认 false；开启后识别 GGUF CONTROL/UNKNOWN。
  USER_DEFINED（例如当前模型的 `<think>`）始终作为原子 token。
- `add_bos`/`add_eos` 为 nullopt 时读取 GGUF 默认值；显式 bool 覆盖默认。
  未配置相应 ID 却要求添加时返回错误。添加是无条件前缀/后缀，不自动去重。
- `decode` 默认隐藏 CONTROL/UNKNOWN，保留 USER_DEFINED；`skip_special=false`
  保留控制标记字面内容。UNUSED token 和越界 ID 返回错误。
- `token_to_piece` 返回原始字节，单个片段可能不是完整 UTF-8；流式显示需在调用方
  拼接完整字符。该函数不隐藏特殊 token，也不做替换字符处理。
- `is_eog` 使用 EOS/EOT/EOM 元数据，以及 Qwen profile 的 `<|endoftext|>`。
  它不负责停止循环，也不把任意特殊 token 当作结束标记。
- 不执行聊天模板；模板格式化与生成循环仍由调用方处理。

GGUF 必需字段：model、pre、tokens、token_type、merges。
词表、类型、merge 和可选 BOS/EOS 设置均在创建时校验。
GGUF 将若干 FIM 标记定义为 CONTROL，而官方 JSON 将其定义为 non-special added
token；本模块遵守 GGUF 类型，因此关闭 parse_special 时这一类标记的行为有差异。

## 构建与测试

需要 PCRE2（8-bit、Unicode 支持）和 ICU（uc）。依赖只出现在内部实现。
不修改根 CMakeLists.txt；提供独立构建入口：

```sh
cmake -S tests/tokenizer -B /tmp/inferllm-tokenizer-build \
  -DCMAKE_PREFIX_PATH=/home/wcf/miniconda3 -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/inferllm-tokenizer-build -j 4
ctest --test-dir /tmp/inferllm-tokenizer-build --output-on-failure
```

系统安装依赖时可省略 CMAKE_PREFIX_PATH。以后接入主程序时，需将本目录四个
`.cpp` 加入构建并链接 `pcre2-8`、`ICU::uc`；当前主程序尚未接入。

参考测试使用 Qwen 官方 tokenizer.json 与 Python `tokenizers` 生成期望值：

```sh
python3 tests/tokenizer/generate_reference.py /path/to/tokenizer.json /tmp/reference.tsv
/tmp/inferllm-tokenizer-build/test_tokenizer_reference \
  models/Qwen3-0.6B-F32.gguf /tmp/reference.tsv
```

Python 和官方 JSON 仅用于测试，C++ 推理不依赖它们。测试包含确定性边界输入、
随机混合文本、特殊 token 开关及解码结果。对照比较完整 ID 序列，不只检查往返。
Unicode 分类与规范化版本由所链接的 PCRE2/ICU 提供；升级依赖后应重跑参考测试。

参考配置：https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer.json
