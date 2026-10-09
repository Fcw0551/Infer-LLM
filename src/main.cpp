#include "../include/gguf_parser.hpp"
#include "../include/model.hpp"
#include "../include/models/Qwen3.hpp"
#include "../include/graph.hpp"
#include <numeric>
int main(){
    auto gguf=GGUFContext::load("/home/wcf/Infer-LLM/models/Qwen3-0.6B-Q8_0.gguf");
    (*gguf).print(10);
    register_all_models();                      //注册模型 
    auto result=ModelBase::load_model(std::move(*gguf));
    if(!result){
        std::cout<<"error"<<result.error()<<std::endl;
        return -1;
    }
    std::unique_ptr<ModelBase> qwen3_ptr=std::move(*result);
    auto qwen3=qwen3_ptr.get();
    qwen3->load_hparams();
    auto result1=qwen3->load_weight();
    if(!result1){
        std::cout<<"error"<<result1.error()<<std::endl;
        return -1;
    }
    
    //要构造一个graph context
    GraphContext gc;
    // 准备 token ids 和 positions
    std::vector<int32_t> token_ids = {151643, 100, 200, 300};
    std::vector<int32_t> positions(token_ids.size());
    std::iota(positions.begin(), positions.end(), 0); // [0,1,2,3]

    // 构造 input_tokens Tensor
    auto inp_tokens = std::make_unique<Tensor>("input_tokens",DataType::GGML_TYPE_I32,std::initializer_list<int64_t>{static_cast<int64_t>(token_ids.size())});
    inp_tokens->op = OperationType::GGML_OP_NONE;
    inp_tokens->data = std::malloc(token_ids.size() * sizeof(int32_t));
    std::memcpy(inp_tokens->data, token_ids.data(),token_ids.size() * sizeof(int32_t));

    //  构造 input_pos Tensor
    auto inp_pos = std::make_unique<Tensor>("input_pos",DataType::GGML_TYPE_I32,std::initializer_list<int64_t>{static_cast<int64_t>(positions.size())});
    inp_pos->op = OperationType::GGML_OP_NONE;
    inp_pos->data = std::malloc(positions.size() * sizeof(int32_t));
    std::memcpy(inp_pos->data, positions.data(),positions.size() * sizeof(int32_t));

    gc.set_input_tokens(inp_tokens.get());
    gc.set_input_pos(inp_pos.get());
    //建图
    qwen3->build_graph(gc);

    return 0;
}


