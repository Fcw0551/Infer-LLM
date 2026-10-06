#include "../include/gguf_parser.hpp"
#include "../include/model.hpp"
int main(){
    auto gguf=GGUFContext::load("/home/wcf/Infer-LLM/models/internlm2-1_8b.q8_0.gguf");
    (*gguf).print(219);
    register_all_models();                      //注册模型 
    auto interlm2=ModelBase::load_model(std::move(*gguf));
    if(!interlm2){
        std::cout<<"error"<<std::endl;
    }
    (*(*interlm2)).load_hparams();
    (*(*interlm2)).load_weight();
    (*(*interlm2)).build_graph();
    return 0;
}


