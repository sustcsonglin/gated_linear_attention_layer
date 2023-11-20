#include <stdio.h>
#include <cuda_runtime.h>
#include "ATen/ATen.h"
#define MIN_VALUE (-1e38)
typedef at::BFloat16 bf16;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void fwd_o_chunk16_dim64x(int batchSize, int M, int N_K, int N_V,
                                     bf16 *Q, bf16 *K, bf16 *V, float *G_K, float *G_V,
                                     float *QK,
                                     bf16 *O
                                    ) {

  // Batch index
  const uint batchIdx = blockIdx.x;

  // allocate buffer for current block in fast shared mem
  __shared__ float Q_tile[32][8];
  __shared__ float K_tile[32][8];
  __shared__ float G_tile[32][8];
  __shared__ float G_tile_trans[8][32];

  const uint threadCol = threadIdx.x % 8;
  const uint threadRow = threadIdx.x / 8;

  int K_Stride = M * N_K;
  int V_Stride = M * N_V;

  // Adjust the pointers for batch and matrix size
  Q += batchIdx * K_Stride;
  K += batchIdx * K_Stride;
  V += batchIdx * V_Stride;
  G_K += batchIdx * K_Stride;
  G_V += batchIdx * V_Stride;  
  O += batchIdx * V_Stride;
  QK += batchIdx * M * M;
  
  float tmp[4] = {0.0};
  
  // we have 256 thread to compute 32x32 matrix = 1024 element=> each thread need compute four element.
  // for loading. we can load 256 element in one time. so, 8 of N_K can be loaded in one time.
  // each thread will compute 8 sum for 4 element they are responsible for.
  for (int bkIdx = 0; bkIdx < N_K; bkIdx += 8) {    
    Q_tile[threadRow][threadCol] = (float)(Q[threadRow * N_K + threadCol]);                
    K_tile[threadRow][threadCol] = (float)(K[threadRow * N_K + threadCol]);
    float tmp_gk = G_K[threadRow * N_K + threadCol];
    G_tile[threadRow][threadCol] = tmp_gk;
    G_tile_trans[threadCol][threadRow] = tmp_gk;
    __syncthreads();

    Q += 8;
    K += 8;
    G_K += 8; 
    
    for(int i = 0; i < 4; i++){
        if (threadRow >= (threadCol * 4 + i)){
                    for (int dotIdx = 0; dotIdx < 8; ++dotIdx) {
                float exp_term = expf(G_tile[threadRow][dotIdx] - G_tile_trans[dotIdx][threadCol * 4 + i]);
                tmp[i] += Q_tile[threadRow][dotIdx] * K_tile[threadCol * 4 + i][dotIdx] * exp_term;
            }
        }
    }
    __syncthreads();    
  }  

  __shared__ float A[32][32];

  for(int i = 0; i < 4; i++){
    A[threadRow][threadCol * 4 + i] = tmp[i];
    QK[threadRow * M + threadCol * 4 + i] = tmp[i];
  }

  __syncthreads(); 

  // we have all elements of A in shared memory. Now we do matrix multiplication.
  const uint B_row = threadIdx.x / 8;
  const uint B_column = (threadIdx.x % 8) ;
  __shared__ float V_tile[32][8];

  __syncthreads();
  
  for(int gg =0; gg < N_V; gg+=8){
        V_tile[B_row][B_column] = float(V[B_row * N_V + B_column]);
        G_tile[B_row][B_column] = G_V[B_row * N_V + B_column];
        __syncthreads();

        float result = 0.0;        
        for(uint dotIdx = 0; dotIdx < 32; dotIdx += 1){
            if (dotIdx <= B_row){
                result += V_tile[dotIdx][B_column] * expf(G_tile[B_row][B_column] - G_tile[dotIdx][B_column]) * A[B_row][dotIdx];
            }
        }         
        O[B_row * N_V + B_column] = result;
        V += 8;
        G_V += 8;
        O += 8;
        __syncthreads();
    }
}







__global__ void compute_d_kernel(int batchSize, int M, int N_K, int N_V,
                                 bf16 *Q, bf16 *K, bf16 *V, float *GK, float *GV,
                                 float *QK,
                                    
                                 bf16 *DO,
                                 bf16 *DQ, bf16 *DK, bf16 *DV, float *DG_K, float *DG_V                                  
                                ) {
    
    const uint batchIdx = blockIdx.x;

    const uint B_row = threadIdx.x / 8;
    const uint B_column = (threadIdx.x % 8) ;

    const uint C_row = threadIdx.x / 8;
    const uint C_column = threadIdx.x % 8; 

    // in fact M must be equal to K. so bStride and cStride should be the same
    int K_Stride = M * N_K;
    int V_Stride = M * N_V; 
    
    Q += batchIdx * K_Stride;
    V += batchIdx * V_Stride;
    K += batchIdx * K_Stride;
    QK += batchIdx * M * M;

    GK += batchIdx * K_Stride;
    GV += batchIdx * V_Stride;
    
    DQ += batchIdx * K_Stride;
    DV += batchIdx * V_Stride;
    DK += batchIdx * K_Stride;
    DG_K += batchIdx * K_Stride;
    DG_V += batchIdx * V_Stride;
    DO += batchIdx * V_Stride;
    
    __shared__ float QK_tile[32][32];

    __shared__ float V_tile[32][8];
    __shared__ float GV_tile[32][8];
    __shared__ float DO_tile[32][8];

    // 32 * 32 element, 256 thread. each thread need 4.
    // 8*i for memory coalescing? yes.
    for(int i = 0; i < 4; i++){
        QK_tile[C_row][C_column + 8*i] = QK[C_row * 32 + 8 * i + C_column];
    }

    float threadResults_dQK[4] = {0.0};    
    __syncthreads();

    for(int gg =0; gg < N_V; gg+=8){
        // float threadResults_dV[4] = {0.0};
        // float threadResults_dgv[4] = {0.0};        
        
        V_tile[B_row][B_column] = float(V[B_row * N_V + B_column]);
        GV_tile[B_row][B_column] = GV[B_row * N_V + B_column];
        DO_tile[B_row][B_column] = float(DO[B_row * N_V + B_column]);
        __syncthreads();

        float threadResults_dV = 0;
        float threadResults_dgv = 0;

        for(uint dotIdx = 0; dotIdx < 32; dotIdx += 1){
            if (dotIdx >= B_row) { 
                float tmp =  DO_tile[dotIdx][B_column] * expf(GV_tile[dotIdx][B_column] - GV_tile[B_row][B_column]) *  QK_tile[dotIdx][B_row];
                threadResults_dV += tmp;
                threadResults_dgv -= tmp * V_tile[B_row][B_column];                
            }
        }
        

        for(uint dotIdx = 0; dotIdx < 32;  dotIdx += 1){
            if (dotIdx <= B_row){
                float tmp = DO_tile[B_row][B_column] * expf(GV_tile[B_row][B_column] - GV_tile[dotIdx][B_column]) * V_tile[dotIdx][B_column];                           
                threadResults_dgv += tmp * QK_tile[B_row][dotIdx];                                            
            }
        }

        if(C_column <= C_row){
            for(int i = 0; i < 4; i ++){
            for(uint resIdx = 0; resIdx < 8; resIdx += 1)
            {
                threadResults_dQK[i] += DO_tile[C_row][resIdx] * expf(GV_tile[C_row][resIdx] - GV_tile[C_column + 8*i][resIdx]) * V_tile[C_column + 8*i][resIdx];
            }
            }
        }
        DV[B_row * N_V + B_column] = bf16(threadResults_dV);        
        DG_V[B_row * N_V + B_column] = threadResults_dgv;
        DV += 8;
        DG_V += 8;
        V += 8;
        GV += 8;        
        DO += 8;         
        __syncthreads();   
    }


    for(int i = 0; i < 4; i++){
        QK_tile[C_row][C_column + 8*i] = threadResults_dQK[i];
        // QK[C_row * 32 + 8 * i + C_column];
    }
    // QK_tile[C_row][C_column] = threadResults_dQK;

    __shared__ float Q_tile[32][8];
    __shared__ float K_tile[32][8];
    __shared__ float GK_tile[32][8];
    __syncthreads();
        
    for(int gg =0; gg < N_K; gg+=8){
        float threadResults_dQ = 0.0;
        float threadResults_dK = 0.0;        
        float threadResults_dgk = 0.0;        

        GK_tile[B_row][B_column] = GK[B_row * N_K + B_column];
        Q_tile[B_row][B_column] = float(Q[B_row * N_K + B_column]);
        K_tile[B_row][B_column] = float(K[B_row * N_K + B_column]);
         
        __syncthreads();

        for(uint dotIdx = 0; dotIdx < 32; dotIdx += 1){
            if (dotIdx >= B_row){
                float tmp =  QK_tile[dotIdx][B_row] * expf(GK_tile[dotIdx][B_column] - GK_tile[B_row][B_column]) * Q_tile[dotIdx][B_column];
                threadResults_dK += tmp;
                threadResults_dgk -= tmp * K_tile[B_row][B_column];                
            }
        }

        for(uint dotIdx = 0; dotIdx < 32;  dotIdx += 1){
            if (dotIdx <= B_row){
                float tmp = QK_tile[B_row][dotIdx] * expf(GK_tile[B_row][B_column] - GK_tile[dotIdx][B_column]) * K_tile[dotIdx][B_column];                                           
                threadResults_dQ += tmp;                       
                threadResults_dgk += tmp * Q_tile[B_row][B_column];
            }
        }

        DQ[B_row * N_K + B_column] = bf16(threadResults_dQ);                
        DK[B_row * N_K + B_column] = bf16(threadResults_dK);  
        DG_K[B_row * N_K + B_column] = threadResults_dgk;
        DQ += 8;
        DK += 8;
        DG_K += 8;
        GK += 8;
        Q += 8;
        K += 8;                
        __syncthreads();

    }
}

void run_fwd_o_chunk16_dim64x(int batchSize, int M, int N_K, int N_V,
                                bf16 *Q, bf16 *K, bf16 *V, 
                                float *gK, float *gV,
                                float *QK,
                                bf16 *O) {  
  dim3 gridDim(batchSize); 
  dim3 blockDim(256);
  fwd_o_chunk16_dim64x<<<gridDim, blockDim>>>(batchSize, M, N_K, N_V, Q, K, V, gK, gV, QK, O); 
}




void run_bwd_o_chunk16_dim64x(int batchSize, int M, int N_K, int N_V,  
                                bf16 *Q, bf16 *K, bf16 *V,
                                float *gK, float *gV, float *QK,
                                bf16 *DO, 
                                bf16 *DQ, bf16 *DK, bf16 *DV,
                                float *DgK, float *DgV
                                ) {
    dim3 gridDim(batchSize); 
    dim3 blockDim(256);  
    compute_d_kernel
    <<<gridDim, blockDim>>>(batchSize, M, N_K, N_V, Q, K, V, gK, gV, QK, DO, DQ, DK, DV, DgK, DgV); 
}




