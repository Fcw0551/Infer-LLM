#include "../../include/models/Qwen3.hpp"
#include "../../include/log.h"

// Qwen3 建计算图
std::expected<Graph, std::error_code> Qwen3Model::build_graph(GraphContext& g_ctx) const{
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
    auto r = g_ctx.get_rope_theta_table(
        _hparams.rope_theta,   // 1000000.0f
        _hparams.n_embd_head,     // 128
        _hparams.n_ctx_train   // 32768
    );
	if(!r){
		return std::unexpected(r.error());
	}
	auto rope_theta_table=*r;

	//attention 所需要的参数
	int64_t n_tokens = inp_pos->dims[0];
	
	AttnParams ap;
	ap.n_head = _hparams.n_head;						      // 16
	ap.n_head_kv = _hparams.n_head_kv;					      // 8
	ap.head_dim = _hparams.n_embd_head;					   	  // 128
	ap.scale = 1.0f / std::sqrt(float(_hparams.n_embd_head)); // 1/sqrt(128)
	ap.causal = true;									      // Prefill 阶段开启
	ap.n_ctx = static_cast<uint32_t>(n_tokens);   		      // 填 token 数(TODO，先简单处理，因为还没有实现kvcache。要区分要算多少token)
	ap.sliding_window = 0;								      // Qwen3 这里不需要，就填 0

	for (size_t i = 0; i < _hparams.n_layer; i++){
		// transfomer
		
		//先保存后续要残差
		Tensor* inpSA=inpL;

		// rms_norm
		Tensor *cur = OpFactory::rms_norm(g, inpL, _weights.layers[i].attn_norm, _hparams.rms_norm_eps, "attn_norm_" + std::to_string(i));

		// x*Wq  x*Wk  x*Wv
		Tensor *q = OpFactory::mul_mat(g, _weights.layers[i].wq, cur, "q_proj" + std::to_string(i));
		Tensor *k = OpFactory::mul_mat(g, _weights.layers[i].wk, cur, "k_proj" + std::to_string(i));
		Tensor *v = OpFactory::mul_mat(g, _weights.layers[i].wv, cur, "v_proj" + std::to_string(i));

		// qk -> rms_norm
		Tensor *q_norm = OpFactory::rms_norm(g, q, _weights.layers[i].attn_q_norm, _hparams.rms_norm_eps, "attn_q_norm_" + std::to_string(i));
		Tensor *k_norm = OpFactory::rms_norm(g, k, _weights.layers[i].attn_k_norm, _hparams.rms_norm_eps, "attn_k_norm_" + std::to_string(i));

		// q_norm k_norm  -->  rope  (TODO:这里可能性能会差，每次都要sin和cos计算，后续可以用空间换时间)
		Tensor *q_rope = OpFactory::rope(g, q_norm, inp_pos, rope_theta_table, "rope_q_" + std::to_string(i));
		Tensor *k_rope = OpFactory::rope(g, k_norm, inp_pos, rope_theta_table, "rope_k_" + std::to_string(i));

		// TODO：kvcache暂时还为做

		// SDPA / FlashAttention
		cur = OpFactory::flash_attn(g, q_rope, k_rope, v, ap, "flash_attn_" + std::to_string(i));

		// Wo
		cur = OpFactory::mul_mat(g, _weights.layers[i].wo, cur, "o_proj" + std::to_string(i));

		// linear
		Tensor *ffn_inp = OpFactory::add(g, cur, inpSA, "ffn_inp_" + std::to_string(i));

		// feed-forward network ffn下一阶段

		// rms_norm
		cur = OpFactory::rms_norm(g, ffn_inp, _weights.layers[i].ffn_norm, _hparams.rms_norm_eps, "ffn_norm_" + std::to_string(i));

		// 两路并行投影
		Tensor *ffn_gate = OpFactory::mul_mat(g, _weights.layers[i].ffn_gate, cur, "ffn_gate_" + std::to_string(i));
		Tensor *ffn_up = OpFactory::mul_mat(g, _weights.layers[i].ffn_up, cur, "ffn_up_" + std::to_string(i));

		// silu
		Tensor *ffn_gate_silu = OpFactory::silu(g, ffn_gate,"ffn_gate_silu_"+std::to_string(i));

		// mul
		Tensor *ffn_inter = OpFactory::mul(g, ffn_up, ffn_gate_silu,"ffn_inter_"+std::to_string(i));

		// mul_mat
		Tensor *ffn_out = OpFactory::mul_mat(g, _weights.layers[i].ffn_down, ffn_inter, "ffn_out_"+std::to_string(i));

		// linear
		cur = OpFactory::add(g, ffn_out, ffn_inp,"result_out_"+std::to_string(i));

		//把cur赋给下一层接着输入
		inpL=cur;
	}

	// n_layer结束之后
	
	//rms_norm
	Tensor *final_norm = OpFactory::rms_norm(g, inpL, _weights.output_norm, _hparams.rms_norm_eps, "result_norm");
	
	Tensor *logits = OpFactory::mul_mat(g, _weights.output, final_norm,"logits");

	// 标记输出
	g.mark_output(logits);

	// 把计算图导成 dot：丢到 https://dreampuf.github.io/GraphvizOnline/ 看，
	// 或本地 `dot -Tsvg qwen3_graph.dot -o qwen3_graph.svg`。
	// 后两个参数是层的闭区间 [lo, hi]：全图 820 个点会把在线渲染器（viz.js 堆只有 16MB）撑爆，
	// 所以这里只导第 0~1 层；想看全图传 (0, INT_MAX)，只想看某几层就改这两个数。
	// 失败只是画不出图，不影响推理，所以只警告不上抛。
	if (auto r = DotWriter::dump(g, "qwen3_graph.dot", 0, 0); !r) {
		std::cerr << "dump graph failed: " << r.error().message() << std::endl;
	}

	std::cout << "qwen3Model build_graph end........" << std::endl;
	return g;
}