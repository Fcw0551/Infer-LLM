#pragma once 
#include "../model.hpp"
//具体的模型去继承ModelBase
class internlm2Model: public ModelBase{
public:
    internlm2Model(GGUFContext&& gguf) 
    : ModelBase(std::move(gguf))
    {}
    virtual std::expected<Graph,std::error_code> build_graph(){
        std::cout<<"internlm2Model build_graph"<<std::endl;
    }
};