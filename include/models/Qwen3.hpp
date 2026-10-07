#pragma once 
#include "../model.hpp"
//具体的模型去继承ModelBase
class Qwen3Model: public ModelBase{
public:
    Qwen3Model(GGUFContext&& gguf) 
    : ModelBase(std::move(gguf))
    {}
    virtual std::expected<void,std::error_code> build_graph();
};