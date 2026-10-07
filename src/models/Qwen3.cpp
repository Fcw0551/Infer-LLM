#include "../../include/models/Qwen3.hpp"
#include "../../include/model.hpp"

//Qwen3 建计算图
virtual std::expected<Graph,std::error_code> Qwen3Model::build_graph(const GraphContext& g_ctx) const{
        std::cout<<"qwen3Model build_graph start........"<<std::endl;
        
        Graph g;                                                        //后面返回出去tensor的生命依旧被引用，只有全部被析构的时候才没
        
        const uint32_t n_embd_head=_hparams.n_embd_head;                //每个头的维度
        
        Tensor* input_token=g_ctx.get_input_tokens();
        Tensor* token_embd=_weights.token_embd;
        embedding(g,input_token,token_embd);                            //embedding

        std::cout<<"qwen3Model build_graph end........"<<std::endl;
}