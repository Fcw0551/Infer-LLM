#include "../include/gguf_parser.h"
int main(){
    auto gguf=GGUFContext::load("/home/wcf/Infer-LLM/models/internlm2-1_8b.q8_0.gguf");
    (*gguf).print(10);
    return 0;
}


