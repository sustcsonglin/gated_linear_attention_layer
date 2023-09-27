#include <torch/extension.h>

void run_fwd_o_chunk16_dim64x(int batchSize, int M, int N,
                                float *Q, float *K, float *V, 
                                float *gK, float *gV,
                                float *QK,
                                float *O);

void run_bwd_o_chunk16_dim64x(int batchSize, int M, int N,  
                                float *Q, float *K, float *V,
                                float *gK, float *gV, float *QK,

                                float *DQ, float *DK, float *DV,
                                float *DgK, float *DgV, 
                                float *DO
                                );                   


std::vector<torch::Tensor> fwd(torch::Tensor Q,
torch::Tensor K, torch::Tensor V, 
torch::Tensor g_K, torch::Tensor g_V
) {   
    auto O = torch::empty_like(Q);
    auto QK = torch::empty({Q.size(0), Q.size(1), Q.size(2), Q.size(3), Q.size(3)}, Q.options());    

    int B_size = Q.size(0); // This is the batch size dimension.
    int H_size = Q.size(1); // This is the head dimension
    int num_chunk = Q.size(2); // This is the chunk dimension.    
    int M = Q.size(-2); // this is the chunk size
    int N = Q.size(-1); // this is the head dim
    
    run_fwd_o_chunk16_dim64x(B_size * H_size * num_chunk, M, N, 
      Q.data_ptr<float>(), K.data_ptr<float>(),  V.data_ptr<float>(), 
      g_K.data_ptr<float>(), g_V.data_ptr<float>(), QK.data_ptr<float>(), O.data_ptr<float>());
      
    return {O, QK};
}




std::vector<torch::Tensor> bwd(torch::Tensor Q,
torch::Tensor K, torch::Tensor V, 
torch::Tensor g_K, torch::Tensor g_V, torch::Tensor QK, torch::Tensor DO
      ) {
    
    auto DQ = torch::empty_like(Q);
    auto DK = torch::empty_like(K);
    auto DV = torch::empty_like(V);
    auto Dg_K = torch::empty_like(g_K);
    auto Dg_V = torch::empty_like(g_V);
    
    int B_size = Q.size(0); // This is the batch size dimension.
    int H_size = Q.size(1); // This is the head dimension
    int num_chunk = Q.size(2); // This is the chunk dimension.    
    int M = Q.size(-2);
    int N = Q.size(-1);

    run_bwd_o_chunk16_dim64x(B_size * H_size * num_chunk, M, N,  
                        Q.data_ptr<float>(),
                        K.data_ptr<float>(), 
                        V.data_ptr<float>(),
                        g_K.data_ptr<float>(),
                        g_V.data_ptr<float>(), 
                        QK.data_ptr<float>(),

                        DO.data_ptr<float>(),
                        DQ.data_ptr<float>(), 
                        DK.data_ptr<float>(),
                        DV.data_ptr<float>(),
                        Dg_K.data_ptr<float>(),
                        Dg_V.data_ptr<float>()
                        );                
    
    return {DQ, DK, DV, Dg_K, Dg_V};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwd, "Batched matrix multiplication with shared memory (CUDA)");
    m.def("backward", &bwd, "Batched matrix multiplication with shared memory (CUDA)");
}



