#pragma once 
#include "../model.hpp"
#include "../graph.hpp"
//具体的模型去继承ModelBase
class internlm2Model: public ModelBase{
public:
    internlm2Model(GGUFContext&& gguf) 
    : ModelBase(std::move(gguf))
    {}
    std::expected<Graph, std::error_code> build_graph(GraphContext& g_ctx) const override {
        return build_decoder_graph(g_ctx, false, true);
    }
};