#include "tensor.hpp"
#include "kernels/cuda/kernel_f32.hpp"
#include "op_params.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
int main() {
    Tensor q("q", GGML_TYPE_F32, {1,2,2,1}), k("k", GGML_TYPE_F32, {1,1,6,1});
    Tensor v("v", GGML_TYPE_F32, {1,1,6,1}), out("out", GGML_TYPE_F32, {1,2,2,1});
    for (auto* t : {&q,&k,&v,&out}) check(cudaMalloc(&t->data,t->byte_size()));
    check(cudaMemset(q.data,0,q.byte_size()));
    check(cudaMemset(k.data,0,k.byte_size()));
    float values[] = {2,7,17,43,111,257};
    check(cudaMemcpy(v.data,values,sizeof(values),cudaMemcpyHostToDevice));
    out.src[0]=&q; out.src[1]=&k; out.src[2]=&v;
    for (unsigned window : {0,1,2,6}) for (unsigned sink : {0,2,5}) {
        AttnParams p{.n_ctx=4,.sliding_window=window,.attention_sink_tokens=sink};
        std::memcpy(out.op_params,&p,sizeof(p));
        check(CUDA_F32::compute_flash_attn(&out,nullptr));
        float actual[4];
        check(cudaMemcpy(actual,out.data,sizeof(actual),cudaMemcpyDeviceToHost));
        for (unsigned row=0; row<2; ++row) {
            const unsigned query=4+row;
            float sum=0;
            unsigned count=0;
            for (unsigned key=0; key<=query; ++key) {
                if (window==0 || key<sink || query-key<window) { sum+=values[key]; ++count; }
            }
            for (unsigned head=0;head<2;++head) {
                if (!(std::abs(actual[row*2+head]-sum/count)<1e-4))
                    throw std::runtime_error("mask output differs from CPU expected mean");
            }
        }
    }
    for (auto* t : {&q,&k,&v,&out}) check(cudaFree(t->data));
    std::cout << "12 mask configurations passed\n";
}
