#include <stdio.h>

#include <type_traits>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

const int K9_NUM_THREADS = 256;

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS)
    fwd_attn_chunk64x(int M, int N, int K,  float *A, float *B,
                   float *G, float *C) {

  const uint batchIdx = blockIdx.x;
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.z;

  // if (cRow < cCol){
  //   return;
  // }
  // size of warptile

  constexpr int WM = TM * 16;
  constexpr int WN = TN * 16;
  // iterations of warptile
  constexpr int WMITER = CEIL_DIV(BM, WM);
  constexpr int WNITER = CEIL_DIV(BN, WN);

  // Placement of the thread in the warptile
  const int threadCol = threadIdx.x % (WN / TN);
  const int threadRow = threadIdx.x / (WN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float GAs[BM * BK];
  __shared__ float Bs[BK * BN];
  __shared__ float GBs[BK * BN];

  // __shared__ float GBs[BK * BN];
  // Move blocktile to beginning of A's row and B's column
  A += batchIdx * M * K + cRow * 64 * K;
  B += batchIdx * K * N + cCol * 64 * K;  
  C += batchIdx * M * N + cRow * 64 * M + cCol * 64;
  
  float * G_A = G + batchIdx * M * K + cRow * 64 * K;
  float * G_B = G + batchIdx * K * N + cCol * 64 * K;
  
  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;

  // const uint innerRowB = threadIdx.x / (BN / 4);
  // const uint innerColB = threadIdx.x % (BN / 4);
  // constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);
  // allocate thread-local cache for results in registerfile

  float threadResults[WMITER * WNITER * TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  float regGM[TM] = {0.0};
  float regGN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // transpose A while storing it
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &B[(innerRowA + offset) * K + innerColA * 4])[0];
      // transpose A while storing it
      Bs[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      Bs[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      Bs[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      Bs[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &G_A[(innerRowA + offset) * K + innerColA * 4])[0];
      // transpose A while storing it
      GAs[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      GAs[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      GAs[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      GAs[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &G_B[(innerRowA + offset) * K + innerColA * 4])[0];
      // transpose A while storing it
      GBs[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      GBs[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      GBs[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      GBs[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    
    __syncthreads();

    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
      for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // block into registers
          for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
          }
          for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }
        
          for (uint i = 0; i < TM; ++i) {
            regGM[i] = GAs[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
          }
          
          for (uint i = 0; i < TN; ++i) {
            regGN[i] = GBs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }

          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                            wnIdx * TN + resIdxN] +=
                  regM[resIdxM] * regN[resIdxN] * ( (regGM[resIdxM] <=  regGN[resIdxN]) ?  expf(regGM[resIdxM] - regGN[resIdxN]) : 0);
            }
          }
        }
      }
    }
    __syncthreads();
    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK; // move BK rows down
    G_A += BK;
    G_B += BK; 
  }

  // write out the results
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          // float4 tmp = reinterpret_cast<float4 *>(
          //     &C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
          //                resIdxN])[0];
          // // perform GEMM update in reg

          const int i =
              (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;
          // tmp.x = threadResults[i + 0];
          // tmp.y = threadResults[i + 1];
          // tmp.z = threadResults[i + 2];
          // tmp.w = threadResults[i + 3];
          // write back
          reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N +
                                                threadCol * TN + resIdxN])[0] =
              reinterpret_cast<float4 *>(&threadResults[i])[0];
        }
      }
    }
  }
  __syncthreads();
}



template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS)
    fwd_o_chunk128(int M, int N, int K, float *A, float *B,
                   float *G, float *C) {
 
  const uint batchIdx = blockIdx.x;
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.z;

  // size of warptile
  constexpr int WM = TM * 16;
  constexpr int WN = TN * 16;
  // iterations of warptile
  constexpr int WMITER = CEIL_DIV(BM, WM);
  constexpr int WNITER = CEIL_DIV(BN, WN);

  // Placement of the thread in the warptile
  const int threadCol = threadIdx.x % (WN / TN);
  const int threadRow = threadIdx.x / (WN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Gs[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column  
  A += batchIdx * M * K + cRow * 64 * K;
  B += batchIdx * K * N + cCol * 64;
  C += batchIdx * M * N + cRow * 64 * N + cCol * 64;
  
  float * G_A = G + batchIdx * M * N + cRow * 64 * N + cCol * 64;
  float * G_B = G + batchIdx * M * N + cCol * 64;
  
  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * WNITER * TM * TN] = {0.0};
  float regGM[WMITER * WNITER * TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};
  float regGN[TN] = {0.0};

  // load all of G_A into registers
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float *C_interim = G_A + (wmIdx * WM * N) + (wnIdx * WN);
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          const int i = (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN; 
          reinterpret_cast<float4 *>(&regGM[i])[0] = reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
        }
      }
    }
  }

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // transpose A while storing it
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }

    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      reinterpret_cast<float4 *>(
          &Gs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &G_B[(innerRowB + offset) * N + innerColB * 4])[0];
    }

    __syncthreads();

    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
      for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // block into registers
          for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
          }
          for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }
          
          for (uint i = 0; i < TN; ++i) {
            regGN[i] = Gs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }

          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              const int idx = (wmIdx * TM + resIdxM) * (WNITER * TN) +
                            wnIdx * TN + resIdxN;
              threadResults[idx] +=
                  regM[resIdxM] * regN[resIdxN] * 
                  ( (regGM[idx] <= regGN[resIdxN]) ? expf(regGM[idx] - regGN[resIdxN]) : 0 );                  
            }
          }
        }
      }
    }

    __syncthreads();
    // advance blocktile
    A += BK;     // move BK columns to right
    G_B += BK * N; 
    B += BK * N; // move BK rows down
  }

  // write out the results
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          // load C vector into registers
          // float4 tmp = reinterpret_cast<float4 *>(
          //     &C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
          //                resIdxN])[0];
          // // perform GEMM update in reg

          const int i =
              (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;
          // tmp.x = threadResults[i + 0];
          // tmp.y = threadResults[i + 1];
          // tmp.z = threadResults[i + 2];
          // tmp.w = threadResults[i + 3];
          // write back
          reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N +
                                                threadCol * TN + resIdxN])[0] =
              reinterpret_cast<float4 *>(&threadResults[i])[0];              
        }
      }
    }
  }
}


// A: attention score, need transpotation. 
// B: gradient
// C: DV
// G, DG: Gate of V, Graident of Gate of V
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS)
    bwd_dv_chunk64x(int M, int N, int K, float *A, float *B, 
                   float *G, float *C, float *V, float *DG){

  // static_assert(M == K, "M should equal to K");

  const uint batchIdx = blockIdx.x;
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.z;

  // size of warptile
  constexpr int WM = TM * 16;
  constexpr int WN = TN * 16;

  // iterations of warptile
  constexpr int WMITER = CEIL_DIV(BM, WM);
  constexpr int WNITER = CEIL_DIV(BN, WN);

  // Placement of the thread in the warptile
  const int threadCol = threadIdx.x % (WN / TN);
  const int threadRow = threadIdx.x / (WN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Gs[BM * BK];
  __shared__ float Bs[BK * BN];

  // __shared__ float Vs[BK * BN];
  // Move blocktile to beginning of A's row and B's column
  // A is transposed. so...
  A += batchIdx * M * K + cRow * 64;

  B += batchIdx * K * N + cCol * 64;

  C += batchIdx * M * N + cRow * 64 * N + cCol * 64;
  
  float * G_B = G + batchIdx * M * N + cRow * 64 * N + cCol * 64;

  V += batchIdx * M * N + cRow * 64 * N + cCol * 64;

  // O += batchIdx * M * N + cRow * 64 * N + cCol * 64;
  // with B

  float * G_A = G + batchIdx * M * N + cCol * 64;     
  DG += batchIdx * M * N + cRow * 64 * N + cCol * 64;

  float* B2 = B + 0;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  
  float regGN[WMITER * WNITER * TM * TN] = {0.0};
  float regV[WMITER * WNITER * TM * TN] = {0.0};

  float regDGN[WMITER * WNITER * TM * TN] = {0.0};
  float regDV[WMITER * WNITER * TM * TN] = {0.0};  
  
  float regM[TM] = {0.0};

  float regN[TN] = {0.0};

  float regGM[TN] = {0.0};

  // load all of G_A into registers
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float *C_interim = G_B + (wmIdx * WM * N) + (wnIdx * WN);
      float *V_interim = V + (wmIdx * WM * N) + (wnIdx * WN);
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          const int i = (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN; 
          reinterpret_cast<float4 *>(&regGN[i])[0] = reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
          reinterpret_cast<float4 *>(&regV[i])[0] = reinterpret_cast<float4 *>(&V_interim[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];                    
        }
      }
    }
  }

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches

      // A need transportation.
      for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &As[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &A[(innerRowB + offset) * M + innerColB * 4])[0];
    }

    // B don't need transportation
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }

    // G是跟着B走的。
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Gs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &G_A[(innerRowB + offset) * N + innerColB * 4])[0];
    }


    // for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    //   float4 tmp = reinterpret_cast<float4 *>(
    //       &V[(innerRowA + offset) * K + innerColA * 4])[0];
    //   // transpose A while storing it
    //   Vs[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    //   Vs[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    //   Vs[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    //   Vs[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    // }

    __syncthreads();


    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
      for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // block into registers
          for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
          }

          for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }
        
          for (uint i = 0; i < TN; ++i) {
            // regGM[i] = Gs[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
            regGM[i] = Gs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }

          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {              

              // const int idx = (wmIdx * TM + resIdxM) * (WNITER * TN) +
              //               wnIdx * TN + resIdxN;                                          
              const int idx = (wmIdx * TM + resIdxM) * (WNITER * TN) +
                            wnIdx * TN + resIdxN;

              float tmp = regM[resIdxM] * regN[resIdxN] * 
                  ((regGM[resIdxN] <= regGN[idx]) ? expf(regGM[resIdxN] - regGN[idx]) : 0);                  
              
              regDV[idx] += tmp;
              regDGN[idx] -= tmp * regV[idx];  
              // * regV[idx];
            }
          }
        }
      }
    }

    __syncthreads();
    
    // advance blocktile

    A += BK * M;     // move BK columns to right
    G_A += BK * N; 
    B += BK * N; // move BK rows down    
  }

  // write out the results
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
      float *DG_interim = DG + (wmIdx * WM * N) + (wnIdx * WN);
      
      float *DO_interim = B2 + (wmIdx * WM * N) + (wnIdx * WN);

      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          const int i =
              (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;

          reinterpret_cast<float4 *>(&DG_interim[(threadRow * TM + resIdxM) * N +
                                                threadCol * TN + resIdxN])[0] =   reinterpret_cast<float4 *>(&regDGN[i])[0]; 

          reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N +
                                                threadCol * TN + resIdxN])[0] =
                reinterpret_cast<float4 *>(&regDV[i])[0]; 

        }
      }
    }
  }
}




// A: attention score, need no transpotation. 
// B: gradient
// C: DV
// G, DG: Gate of V, Graident of Gate of V
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS)
    bwd_dq_chunk64x(int M, int N, int K, float *A, float *B, 
                   float *G, float *C, float *V, float *DG){

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.z;

  const uint batchIdx = blockIdx.x;

  // size of warptile
  constexpr int WM = TM * 16;
  constexpr int WN = TN * 16;
  // iterations of warptile
  constexpr int WMITER = CEIL_DIV(BM, WM);
  constexpr int WNITER = CEIL_DIV(BN, WN);

  // Placement of the thread in the warptile
  const int threadCol = threadIdx.x % (WN / TN);
  const int threadRow = threadIdx.x / (WN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Gs[BM * BK];
  __shared__ float Bs[BK * BN];

  // __shared__ float Vs[BK * BN];
  // Move blocktile to beginning of A's row and B's column
  A += batchIdx * M * K + cRow * 64 * K;
  B += batchIdx * K * N + cCol * 64;
  
  C += batchIdx * M * N + cRow * 64 * N + cCol * 64;
  
  float* G_A = G + batchIdx * M * N + cRow * 64 * N + cCol * 64;
  float* G_B = G + batchIdx * M * N + cCol * 64;

  V += batchIdx * M * N + cRow * 64 * N + cCol * 64; 
  DG += batchIdx * M * N + cRow * 64 * N + cCol * 64;

  float* B2 = B + 0;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  
  float regGN[WMITER * WNITER * TM * TN] = {0.0};
  float regV[WMITER * WNITER * TM * TN] = {0.0};

  float regDGN[WMITER * WNITER * TM * TN] = {0.0};
  float regDV[WMITER * WNITER * TM * TN] = {0.0};  
  
  float regM[TM] = {0.0};

  float regN[TN] = {0.0};

  float regGM[TN] = {0.0};

  // load all of G_A into registers
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float *C_interim = G_A + (wmIdx * WM * N) + (wnIdx * WN);
      float *V_interim = V + (wmIdx * WM * N) + (wnIdx * WN);
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          const int i = (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN; 
          reinterpret_cast<float4 *>(&regGN[i])[0] = reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
          reinterpret_cast<float4 *>(&regV[i])[0] = reinterpret_cast<float4 *>(&V_interim[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];                    
        }
      }
    }
  }

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // A need no transportation.
 
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // transpose A while storing it
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }

    //
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {      

      reinterpret_cast<float4 *>(
          &Gs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &G_B[(innerRowB + offset) * N + innerColB * 4])[0];
    }

    // for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    //   float4 tmp = reinterpret_cast<float4 *>(
    //       &V[(innerRowA + offset) * K + innerColA * 4])[0];
    //   // transpose A while storing it
    //   Vs[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    //   Vs[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    //   Vs[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    //   Vs[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    // }

    __syncthreads();

    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
      for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // block into registers
          for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
          }
          for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }
          for (uint i = 0; i < TN; ++i) {
            // regGM[i] = Gs[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
            regGM[i] = Gs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {              
              // const int idx = (wmIdx * TM + resIdxM) * (WNITER * TN) +
              //               wnIdx * TN + resIdxN;                                          
              const int idx = (wmIdx * TM + resIdxM) * (WNITER * TN) +
                            wnIdx * TN + resIdxN;
              float tmp = regM[resIdxM] * regN[resIdxN] * 
                  (( regGN[idx] <= regGM[resIdxN]) ? expf(regGN[idx] - regGM[resIdxN]) : 0);                  
              
              regDV[idx] += tmp;
              regDGN[idx] += tmp * regV[idx];              
            }
          }
        }
      }
    }

    __syncthreads();
    // advance blocktile
    A += BK;     // move BK columns to right
    G_B += BK * N; 
    B += BK * N; // move BK rows down    
  }


  // write out the results
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
      float *DG_interim = DG + (wmIdx * WM * N) + (wnIdx * WN);
      
      // float *O_interim = O + (wmIdx * WM * N) + (wnIdx * WN);
      // float *DO_interim = B2 + (wmIdx * WM * N) + (wnIdx * WN);

      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {

          // load C vector into registers
          // // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &DG_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
                         resIdxN])[0];

          // // // perform GEMM update in reg

          // float4 tmp2 = reinterpret_cast<float4 *>(
          //     &DO_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
          //                resIdxN])[0];

          const int i =
              (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;

          tmp.x = tmp.x + regDGN[i + 0];
          tmp.y = tmp.y + regDGN[i + 1];
          tmp.z = tmp.z + regDGN[i + 2];
          tmp.w = tmp.w + regDGN[i + 3];

          // tmp.x = threadResults[i + 0];
          // tmp.y = threadResults[i + 1];
          // tmp.z = threadResults[i + 2];
          // tmp.w = threadResults[i + 3];
          // write back

          reinterpret_cast<float4 *>(&DG_interim[(threadRow * TM + resIdxM) * N +
                                                threadCol * TN + resIdxN])[0] = tmp;

          reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N +
                                                threadCol * TN + resIdxN])[0] =
                reinterpret_cast<float4 *>(&regDV[i])[0]; 

        }
      }
    }
  }
}




void run_fwd_attn_chunk64x_dim64x(int batchSize, int M, int N_K,
                                 float *Q, float *K,
                                 float *gK,
                                 float *QK                                
                              ) {  

  // A100
  const uint K9_BK = 16;
  const uint K9_TM = 4;
  const uint K9_TN = 4;
  const uint K9_BM = 64;
  const uint K9_BN = 64;
  const uint K9_NUM_THREADS = 256;

  // // A6000
  // const uint K9_BK = 16;
  // const uint K9_TM = 8;
  // const uint K9_TN = 8;
  // const uint K9_BM = 128;
  // const uint K9_BN = 128;
  // const uint K9_NUM_THREADS = 256;

  dim3 blockDim(K9_NUM_THREADS);    
  static_assert(
      (K9_NUM_THREADS * 4) % K9_BK == 0,
      "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of Bs "
      "during each iteraion)");
  static_assert(
      (K9_NUM_THREADS * 4) % K9_BN == 0,
      "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of As "
      "during each iteration)");
  static_assert(
      K9_BN % (16 * K9_TN) == 0,
      "K9_BN must be a multiple of 16*K9_TN to avoid quantization effects");
  static_assert(
      K9_BM % (16 * K9_TM) == 0,
      "K9_BM must be a multiple of 16*K9_TM to avoid quantization effects");
  static_assert((K9_BM * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BM*K9_BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K9_BN * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BN*K9_BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(batchSize, M / K9_BM, M / K9_BN);
  
  fwd_attn_chunk64x<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN>
  <<<gridDim, blockDim>>>(M, M, N_K, Q, K, gK, QK); 
  


}

void run_bwd_o_chunk64x_dim64x(int batchSize, int M, int N_K,                                  float *DQK, float *Q, float *K, float *gK, 
float *DQ, float *DK, float *DgK
                                ){
  // A100
  const uint K9_BK = 16;
  const uint K9_TM = 4;
  const uint K9_TN = 4;
  const uint K9_BM = 64;
  const uint K9_BN = 64;
  const uint K9_NUM_THREADS = 256;

  dim3 blockDim(K9_NUM_THREADS);

  static_assert(
      (K9_NUM_THREADS * 4) % K9_BK == 0,
      "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of Bs "
      "during each iteraion)");
  
  static_assert(
      (K9_NUM_THREADS * 4) % K9_BN == 0,
      "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of As "
      "during each iteration)");
  
  static_assert(
      K9_BN % (16 * K9_TN) == 0,
      "K9_BN must be a multiple of 16*K9_TN to avoid quantization effects");  
  
  static_assert(
      K9_BM % (16 * K9_TM) == 0,
      "K9_BM must be a multiple of 16*K9_TM to avoid quantization effects");

  static_assert((K9_BM * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BM*K9_BK must be a multiple of 4*256 to vectorize loads");
  
  static_assert((K9_BN * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BN*K9_BK must be a multiple of 4*256 to vectorize loads");
  

  dim3 gridDim3(batchSize, M / 64, N_K / 64);
  bwd_dv_chunk64x<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN>
  <<<gridDim3, blockDim>>>(M, N_K, M, DQK, Q, gK, DK, K, DgK); 

  dim3 gridDim4(batchSize, M / 64, N_K / 64);
  bwd_dq_chunk64x<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN>
  <<<gridDim4, blockDim>>>(M, N_K, M, DQK, K, gK, DQ, Q, DgK); 

}             

  






