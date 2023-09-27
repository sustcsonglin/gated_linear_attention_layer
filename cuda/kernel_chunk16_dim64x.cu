#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void fwd_o_chunk16_dim64x(int batchSize, int M, int N,
                                     float *Q, float *K, float *V, float *G_K, float *G_V,
                                     float *QK,
                                     float *O
                                    ) {

  // Batch index
  const uint batchIdx = blockIdx.x;

  // allocate buffer for current block in fast shared mem
  __shared__ float Q_tile[16][16];
  __shared__ float K_tile[16][16];
  __shared__ float G_tile[16][16];
  __shared__ float G_tile_trans[16][16];

  const uint threadCol = threadIdx.x % 16;
  const uint threadRow = threadIdx.x / 16;

  int aStride = M * N;

  // Adjust the pointers for batch and matrix size
  Q += batchIdx * aStride;
  K += batchIdx * aStride;
  V += batchIdx * aStride;
  G_K += batchIdx * aStride;
  G_V += batchIdx * aStride;  
  O += batchIdx * aStride;
  QK += batchIdx * M * M;
  
  float tmp = 0.0;

  for (int bkIdx = 0; bkIdx < N; bkIdx += 16) {
    Q_tile[threadRow][threadCol] = Q[threadRow * N + threadCol];
    K_tile[threadRow][threadCol] = K[threadRow * N + threadCol];
    float tmp_gk = G_K[threadRow * N + threadCol];
    G_tile[threadRow][threadCol] = tmp_gk;
    G_tile_trans[threadCol][threadRow] = tmp_gk;

    __syncthreads();

    Q += 16;
    K += 16;
    G_K += 16;
    
    if(threadCol <= threadRow){
        for (int dotIdx = 0; dotIdx < 16; ++dotIdx) {
            // avoid bank conflict?
            float exp_term = expf(G_tile[threadRow][dotIdx] - G_tile_trans[dotIdx][threadCol]);
            tmp += Q_tile[threadRow][dotIdx] * K_tile[threadCol][dotIdx] * exp_term;
        }
    }
    __syncthreads();    
  }  

  // to save share memory, Q_tile from now on would be the intermediate result of QK, and K_tile would be V_tile

  if(threadCol <= threadRow){
    Q_tile[threadRow][threadCol] = tmp;
    QK[threadRow * M + threadCol] = tmp;
  }

  __syncthreads(); 

  int num_loop = int(N / 64);

  const uint B_row = threadIdx.x / 16;
  const uint B_column = (threadIdx.x % 16) * 4;

  
  __shared__ float V_tile[16][64];
  __shared__ float G_tile2[16][64];
  __syncthreads();
  
  for(int gg =0; gg < num_loop; gg+=1){
        float threadResults[4] = {0.0};        
        FLOAT4(V_tile[B_row][B_column]) = FLOAT4(V[B_row * N + B_column]);        
        float4 tmp = FLOAT4(G_V[B_row * N + B_column]);
        FLOAT4(G_tile2[B_row][B_column]) = tmp;
        float tmp_g[4] = {0}; 
        FLOAT4(tmp_g[0]) = tmp;        

        __syncthreads();

        for(uint dotIdx = 0; dotIdx <= B_row; dotIdx += 1){
            for(uint resIdx = 0; resIdx < 4; resIdx +=1){
                threadResults[resIdx] += V_tile[dotIdx][B_column + resIdx] * expf(tmp_g[resIdx] - G_tile2[dotIdx][B_column + resIdx]) * Q_tile[B_row][dotIdx];
            }         
        }

        FLOAT4(O[B_row * N + B_column]) = FLOAT4(threadResults[0]);                
        V += 64;
        G_V += 64;
        O += 64; 
        __syncthreads();
    }
}



__global__ void compute_d_kernel(int batchSize, int M, int N,
                                 float *Q, float *K, float *V, float *GK, float *GV,
                                 float *QK,
                                    
                                 float *DO,
                                 float *DQ, float *DK, float *DV, float *DG_K, float *DG_V                                  
                                ) {
    
    const uint batchIdx = blockIdx.x;

    const uint B_row = threadIdx.x / 16;
    const uint B_column = (threadIdx.x % 16) * 4;

    const uint C_row = threadIdx.x / 16;
    const uint C_column = threadIdx.x % 16; 

    // in fact M must be equal to K. so bStride and cStride should be the same
    int aStride = M * N;

    int num_loop = int(N / 64);
    
    
    Q += batchIdx * aStride;
    V += batchIdx * aStride;
    K += batchIdx * aStride;
    QK += batchIdx * M * M;
    GK += batchIdx * aStride;
    GV += batchIdx * aStride;
    
    DQ += batchIdx * aStride;
    DV += batchIdx * aStride;
    DK += batchIdx * aStride;
    DG_K += batchIdx * aStride;
    DG_V += batchIdx * aStride;
    DO += batchIdx * aStride;
    
    __shared__ float QK_tile[16][16];

    __shared__ float V_tile[16][64];
    __shared__ float GV_tile[16][64];
    __shared__ float DO_tile[16][64];

    QK_tile[C_row][C_column] = QK[C_row * 16 + C_column];

    float threadResults_dQK = 0.0;
    
    __syncthreads();

    for(int gg =0; gg < num_loop; gg+=1){
        float threadResults_dV[4] = {0.0};
        float threadResults_dgv[4] = {0.0};        
        

        FLOAT4(V_tile[B_row][B_column]) = FLOAT4(V[B_row * N + B_column]);
        FLOAT4(GV_tile[B_row][B_column]) = FLOAT4(GV[B_row * N + B_column]);
        FLOAT4(DO_tile[B_row][B_column]) = FLOAT4(DO[B_row * N + B_column]);

        __syncthreads();

        for(uint dotIdx = B_row; dotIdx < 16; dotIdx += 1){
            for(uint resIdx = 0; resIdx < 4; resIdx +=1){
                float tmp =  DO_tile[dotIdx][B_column + resIdx] * expf(GV_tile[dotIdx][B_column + resIdx] - GV_tile[B_row][B_column + resIdx]) *  QK_tile[dotIdx][B_row];

                threadResults_dV[resIdx] += tmp;
                threadResults_dgv[resIdx] -= tmp * V_tile[B_row][B_column + resIdx] 
                ;                
            }         
        }

        __syncthreads();

        for(uint dotIdx = 0; dotIdx <= B_row;  dotIdx += 1){
            for(uint resIdx = 0; resIdx < 4; resIdx += 1)
            {
                float tmp = DO_tile[B_row][B_column + resIdx] * expf(GV_tile[B_row][B_column + resIdx] - GV_tile[dotIdx][B_column + resIdx]) * V_tile[dotIdx][B_column + resIdx];                           
                threadResults_dgv[resIdx] +=  tmp * QK_tile[B_row][dotIdx];                                
            }
        }

        if(C_column <= C_row){
            for(uint resIdx = 0; resIdx < 64; resIdx += 1)
            {
                threadResults_dQK += DO_tile[C_row][resIdx] * expf(GV_tile[C_row][resIdx] - GV_tile[C_column][resIdx]) * V_tile[C_column][resIdx];
            }
        }

        __syncthreads();

        FLOAT4(DV[B_row * N + B_column]) = FLOAT4(threadResults_dV[0]);        
        FLOAT4(DG_V[B_row * N + B_column]) = FLOAT4(threadResults_dgv[0]);        
        __syncthreads();

        DV += 64;
        DG_V += 64;
        V += 64;
        GV += 64;
        
        DO += 64;
            
    }

    QK_tile[C_row][C_column] = threadResults_dQK;
    __shared__ float Q_tile[16][64];
    __shared__ float K_tile[16][64];
    __shared__ float GK_tile[16][64];
    __syncthreads();
        
    for(int gg =0; gg < num_loop; gg+=1){
        float threadResults_dQ[4] = {0.0};
        float threadResults_dK[4] = {0.0};        
        float threadResults_dgk[4] = {0.0};        

        FLOAT4(GK_tile[B_row][B_column]) = FLOAT4(GK[B_row * N + B_column]);
        FLOAT4(Q_tile[B_row][B_column]) = FLOAT4(Q[B_row * N + B_column]);
        FLOAT4(K_tile[B_row][B_column]) = FLOAT4(K[B_row * N + B_column]);
         
        __syncthreads();

        for(uint dotIdx = B_row; dotIdx < 16; dotIdx += 1){
            for(uint resIdx = 0; resIdx < 4; resIdx +=1){
                float tmp =  QK_tile[dotIdx][B_row] * expf(GK_tile[dotIdx][B_column + resIdx] - GK_tile[B_row][B_column + resIdx]) * Q_tile[dotIdx][B_column + resIdx];

                threadResults_dK[resIdx] += tmp;

                threadResults_dgk[resIdx] -= tmp * K_tile[B_row][B_column + resIdx];                
            }         
        }

        __syncthreads();

        for(uint dotIdx = 0; dotIdx <= B_row;  dotIdx += 1){
            for(uint resIdx = 0; resIdx < 4; resIdx += 1)
            {
                float tmp = QK_tile[B_row][dotIdx] * expf(GK_tile[B_row][B_column + resIdx] - GK_tile[dotIdx][B_column + resIdx]) * K_tile[dotIdx][B_column + resIdx];                           
                
                threadResults_dQ[resIdx] += tmp;                       
                threadResults_dgk[resIdx] += tmp * Q_tile[B_row][B_column + resIdx];
            }
        }

        FLOAT4(DQ[B_row * N + B_column]) = FLOAT4(threadResults_dQ[0]);                
        FLOAT4(DK[B_row * N + B_column]) = FLOAT4(threadResults_dK[0]);  
        FLOAT4(DG_K[B_row * N + B_column]) = FLOAT4(threadResults_dgk[0]);
                
        __syncthreads();
        DQ += 64;
        DK += 64;
        DG_K += 64;
        GK += 64;
        Q += 64;
        K += 64;
        
    }
}



void run_fwd_o_chunk16_dim64x(int batchSize, int M, int N,
                                float *Q, float *K, float *V, 
                                float *gK, float *gV,
                                float *QK,
                                float *O) {  
  dim3 gridDim(batchSize); 
  dim3 blockDim(256);
  fwd_o_chunk16_dim64x<<<gridDim, blockDim>>>(batchSize, M, N, Q, K, V, gK, gV, QK, O); 
}




void run_bwd_o_chunk16_dim64x(int batchSize, int M, int N,  
                                float *Q, float *K, float *V,
                                float *gK, float *gV, float *QK,
                                float *DO, 
                                float *DQ, float *DK, float *DV,
                                float *DgK, float *DgV
                                ) {
    dim3 gridDim(batchSize); 
    dim3 blockDim(256);  
    compute_d_kernel
    <<<gridDim, blockDim>>>(batchSize, M, N,  Q, K, V, gK, gV, QK, DO, DQ, DK, DV, DgK, DgV); 
}




