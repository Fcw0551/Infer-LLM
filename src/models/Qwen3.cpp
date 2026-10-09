#include "../../include/models/Qwen3.hpp"
#include "../../include/model.hpp"

// Qwen3 建计算图
std::expected<Graph, std::error_code> Qwen3Model::build_graph(GraphContext &g_ctx) const override{
	std::cout << "qwen3Model build_graph start........" << std::endl;

	Graph g; // 后面返回出去tensor的生命依旧被引用，只有全部被析构的时候才没

	const uint32_t n_embd_head = _hparams.n_embd_head; // 每个头的维度

	// 输入
	Tensor *input_token = g_ctx.get_input_tokens();
	Tensor *inp_pos = g_ctx.get_input_pos();
	g.mark_input(input_token);
	g.mark_input(inp_pos);

	Tensor *token_embd = _weights.token_embd;
	Tensor *inpL = OpFactory::embedding(g, input_token, token_embd); // embedding  inpL=input to the layer(当前层输入)

	//rope表
    Tensor* rope_theta_table = g_ctx.get_rope_theta_table(
        _hparams.rope_theta,   // 1000000.0f
        _hparams.head_dim,     // 128
        _hparams.n_ctx_train   // 32768
    );

	//attention 所需要的参数
	int64_t n_tokens = inp_pos->ne[0];
	
	AttnParams ap;
	ap.n_head = _hparams.n_head;						   // 16
	ap.n_head_kv = _hparams.n_head_kv;					   // 8
	ap.head_dim = _hparams.head_dim;					   // 128
	ap.scale = 1.0f / std::sqrt(float(_hparams.head_dim)); // 1/sqrt(128)
	ap.causal = true;									   // Prefill 阶段开启
	ap.n_ctx = static_cast<uint32_t>(n_tokens);   		   // 填 token 数(TODO，先简单处理，因为还没有实现kvcache。要区分要算多少token)
	ap.sliding_window = 0;								   // Qwen3 这里不需要，就填 0

	for (size_t i = 0; i < _hparams.n_layer; i++){
		// transfomer
		
		//先保存后续要残差
		Tensor* inpSA=inpL;
		
		//rms_norm
		Tensor *cur=OpFactory::rms_norm(g, inpL, _weights.layer[i].attn_norm, _hparams.rms_norm_eps);

		// x*Wq  x*Wk  x*Wv
    	Tensor* q=OpFactory::mul_mat(g, _weights.layer[i].wq, cur);
    	Tensor* k=OpFactory::mul_mat(g, _weights.layer[i].wk, cur);
    	Tensor* v=OpFactory::mul_mat(g, _weights.layer[i].wv, cur);

		//qk -> rms_norm
		Tensor *q_norm=OpFactory::rms_norm(g, q, _weights.layer[i].attn_q_norm, _hparams.rms_norm_eps);
		Tensor *k_norm=OpFactory::rms_norm(g, k, _weights.layer[i].attn_k_norm, _hparams.rms_norm_eps);

		//q_norm k_norm  -->  rope  (TODO:这里可能性能会差，每次都要sin和cos计算，后续可以用空间换时间)
		Tensor *q_rope = OpFactory::rope(g, q_norm, inp_pos, rope_theta_table);
		Tensor *k_rope = OpFactory::rope(g, k_norm, inp_pos, rope_theta_table);

		//TODO：kvcache暂时还为做

		//SDPA / FlashAttention
		cur = OpFactory::flash_attn(g, q_rope, k_rope, v, ap);

		// Wo
		cur = OpFactory::mul_mat(g, _weights.layer[i].wo, cur);

		// linear
		Tensor *ffn_inp = OpFactory::add(g, cur, inpSA);

		//feed-forward network ffn下一阶段
		
		//rms_norm
		cur = OpFactory::rms_norm(g, ffn_inp, _weights.layer[i].ffn_norm, _hparams.rms_norm_eps);

		// 两路并行投影
		Tensor *ffn_gate = OpFactory::mul_mat(g, _weights.layers[i].ffn_gate, cur);
		Tensor *ffn_up = OpFactory::mul_mat(g, _weights.layers[i].ffn_up, cur);

		// silu
		Tensor *ffn_gate_silu = OpFactory::silu(g, ffn_gate);

		// mul
		Tensor *ffn_inter = OpFactory::mul(g, ffn_up, ffn_gate_silu);

		// mul_mat
		Tensor *ffn_out = OpFactory::mul_mat(g, _weights.layers[i].ffn_down, ffn_inter);

		// linear
		cur = OpFactory::add(g, ffn_out, ffn_inp);

		//把cur赋给下一层接着输入
		inpL=cur;
	}

	// n_layer结束之后
	
	//rms_norm
	Tensor *final_norm = OpFactory::rms_norm(g, inpL, _weights.output_norm, _hparams.rms_norm_eps);
	
	Tensor *logits = OpFactory::mul_mat(g, _weights.output, final_norm);

	// 标记输出
	g.mark_output(logits);
	std::cout << "qwen3Model build_graph end........" << std::endl;
	return g;
}