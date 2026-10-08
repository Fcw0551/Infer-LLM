#include "../include/gguf_parser.hpp"
#include "../include/model.hpp"
#include "../include/graph.hpp"
int main(){
    auto gguf=GGUFContext::load("/home/wcf/Infer-LLM/models/Qwen3-0.6B-Q8_0.gguf");
    (*gguf).print(219);
    register_all_models();                      //注册模型 
    auto result=ModelBase::load_model(std::move(*gguf));
    if(!result){
        std::cout<<"error"<<result.error()<<std::endl;
    }
    std::unique_ptr<ModelBase> qwen3_ptr=*result;
    Qwen3Model qwen3=*qwen3_ptr;
    qwen3.load_hparams();
    qwen3.load_weight();
    
    //要构造一个graph context
    GraphContext gc;
    //缓存rpoe表
    gc.get_rope_theta_table(qwen3._hparams.rope_theta,qwen3._hparams.n_embd_head,qwen3._hparams.n_ctx_train);

    //建图
    qwen3.build_graph(gc);

    return 0;
}


