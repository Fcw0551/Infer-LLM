#include "../include/gguf_parser.hpp"
int main(){
    auto gguf=GGUFContext::load("/home/wcf/Infer-LLM/models/internlm2-1_8b.q8_0.gguf");
    (*gguf).print(219);
    ModelBase::load_model(std::move(*gguf));
    return 0;
}


