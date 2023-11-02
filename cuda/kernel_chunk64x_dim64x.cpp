#include <torch/extension.h>

void run_fwd_attn_chunk64x_dim64x(int batchSize, int M, int N_K, 
                                  float *Q, float *K,  
                                  float *gK,
                                  float *QK
                                );

void run_bwd_o_chunk64x_dim64x(int batchSize, int M, int N_K,   
                                float *DQK, 
                                float *Q, float *K,
                                float *gK,                           
                                float *DQ, float *DK, 
                                float *DgK 
                                );                   

torch::Tensor fwd(torch::Tensor Q,
torch::Tensor K,  
torch::Tensor g_K
) {   

    auto QK = torch::zeros({Q.size(0), Q.size(1), Q.size(2), Q.size(3), Q.size(3)}, Q.options());    

    int B_size = Q.size(0); // This is the batch size dimension.
    int H_size = Q.size(1); // This is the head dimension
    int num_chunk = Q.size(2); // This is the chunk dimension.    
    int M = Q.size(-2); // this is the chunk size
    int N_K = Q.size(-1); // this is the head_K dim
    
    run_fwd_attn_chunk64x_dim64x(B_size * H_size * num_chunk, M, N_K, 
      Q.data_ptr<float>(), K.data_ptr<float>(),   
      g_K.data_ptr<float>(), QK.data_ptr<float>());
      
    return QK;
}



std::vector<torch::Tensor> bwd(torch::Tensor Q,
torch::Tensor K,  
torch::Tensor g_K,  torch::Tensor DQK
      ) {
    
    auto DQ = torch::zeros_like(Q);
    auto DK = torch::zeros_like(K);
    auto Dg_K = torch::zeros_like(g_K);
    
    int B_size = Q.size(0); // This is the batch size dimension.
    int H_size = Q.size(1); // This is the head dimension
    int num_chunk = Q.size(2); // This is the chunk dimension.    
    int M = Q.size(-2);
    int N_K = Q.size(-1);
    
    run_bwd_o_chunk64x_dim64x(B_size * H_size * num_chunk, M, N_K,    
                        DQK.data_ptr<float>(),
                        Q.data_ptr<float>(),
                        K.data_ptr<float>(), 
                        g_K.data_ptr<float>(),
                        DQ.data_ptr<float>(), 
                        DK.data_ptr<float>(),
                        Dg_K.data_ptr<float>()
                        );                
    return {DQ, DK, Dg_K};
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwd, "Batched matrix multiplication with shared memory chunk64x dim64x (CUDA)");
    m.def("backward", &bwd, "Batched matrix multiplication with shared memory (CUDA)");
}


