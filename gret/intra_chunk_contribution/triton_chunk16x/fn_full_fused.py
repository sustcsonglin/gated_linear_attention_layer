import torch 
import time
import math
from typing import Tuple, Union, Optional


import torch
import torch.nn.functional as F
from einops import rearrange

import torch

import triton
import triton.language as tl

import numpy as np
import math

# from ..cuda_chunk16_dim64x.fn import cuda_compute_intra
# we store the attention matrix cuz our impl uses chunkwise computation anyway. say the chunk size is 64, the attention matrix is only of size 64 * 64. In standard flashattn, this could be very large, say 2048 * 2048. So we can save a lot of recomputation here by saving the attention matrix.
@triton.jit
def _fwd_kernel(
    Q, K, V, GK, GV, 
    O, A, Q_GV, Q_GK,
    stride_q1, stride_q2, stride_q3, stride_q4,
    stride_v1, stride_v2, stride_v3, stride_v4,
    stride_a1, stride_a2, stride_a3, stride_a4,
    Z, H, N_CTX, D,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr,
):
    # Q block
    start_m = tl.program_id(0)
    # Batch & Head 
    off_hz = tl.program_id(1)

    qk_offset = off_hz * stride_q2
    v_offset = off_hz * stride_v2
    a_offset = off_hz * stride_a2 
    
    # initialize pointer to m and l
    
    # load q: it will stay in SRAM throughout
    # loop over k, v and update accumulator
    
    lo = 0
    hi = BLOCK_N 

    Q_ptr = Q + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3
    
    K_ptr = K + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q3

    V_ptr = V + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    GK_K_ptr = GK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q3

    GK_Q_ptr = GK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    GV_ptr = GV + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    A_ptr = A + a_offset + (start_m * BLOCK_M) * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a3

    Q_GK_ptr = Q_GK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3
    
    Q_GV_ptr = Q_GV + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3


    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q3)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q3).to(tl.float32)
        
        q_normalizer = tl.load(GK + qk_offset + (start_m * BLOCK_N + q_high) * stride_q3 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)
        q_gk = tl.exp(q_gk - q_normalizer[None, :])
        q = q * q_gk.to(q.dtype)

        # save for backprop.
        tl.store(Q_GK_ptr + q_high * stride_q3, q_gk.to(Q_GK.dtype.element_ty))

        q_gv_normalizer = tl.load(GV + v_offset + (start_m * BLOCK_N + q_high) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)

        acc = tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32)

        # inter-chunk, bf16
        for k_high in range(0, q_high, 16):
            k = tl.load(K_ptr + k_high * stride_q3)
            k_gk = tl.load(GK_K_ptr + k_high * stride_q3)           
            k_gk = tl.exp(q_normalizer[:, None] - k_gk)
            k = k * k_gk.to(k.dtype)
            
            #bf16            
            qk = tl.dot(q, k, allow_tf32=False)            
            tl.store(A_ptr + q_high * stride_a3 + k_high, qk.to(A_ptr.dtype.element_ty))

            v = tl.load(V_ptr + k_high * stride_v3) 
            k_gv = tl.load(GV_ptr + k_high * stride_v3)
            k_gv = tl.exp(q_gv_normalizer[None, :] - k_gv)
            v = v * k_gv.to(v.dtype)            
            output = tl.dot(qk.to(v.dtype), v, allow_tf32=False)        
            acc += output

        q_gv = tl.load(GV_ptr + q_high * stride_v3).to(tl.float32)
        q_gv = tl.exp(q_gv - q_gv_normalizer[None, :])

        ## for backprop. lots of recomputation.
        tl.store(Q_GV_ptr + q_high * stride_v3, q_gv.to(Q_GV.dtype.element_ty))

        acc = acc * q_gv
        
        O_block_ptr = tl.make_block_ptr(
            base= O + v_offset,
            shape=(N_CTX, BLOCK_DMODEL_V),
            strides=(stride_v3, stride_v4),
            offsets=(start_m * BLOCK_N + q_high, 0),
            block_shape=(16, BLOCK_DMODEL_V),
            order=(1, 0)
        )        
        tl.store(O_block_ptr, acc.to(O.dtype.element_ty))    

    # make compiler easy to generate code
    tl.debug_barrier()

    # seem like a compiler bug. (Segmentation fault (core dumped) if put the following code in the previous loop)
    # so i use a separate for loop below. Not sure about the performance impact. But seems small.


    for q_high in range(lo, hi, 16):
        # q = tl.load(Q_ptr + q_high * stride_q3)
        # q_gk = tl.load(GK_Q_ptr + q_high * stride_q3).to(tl.float32)
        
        # q_gk = tl.exp(q_gk - q_normalizer[None, :])
        # q = q * q_gk.to(q.dtype)
        
        q = tl.load(Q_GK_ptr + q_high * stride_q3).to(tl.float32)
        q_gv_normalizer = tl.load(GV + v_offset + (start_m * BLOCK_N + q_high) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)
        q_normalizer = tl.load(GK + qk_offset + (start_m * BLOCK_N + q_high) * stride_q3 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)

        # intra-chunk, fp32
        k2 = tl.load(K_ptr + q_high * stride_q3)
        k_gk2 = tl.load(GK_K_ptr + q_high * stride_q3)           
        k_gk2 = tl.exp(q_normalizer[:, None] - k_gk2)
        k2 = k2 * k_gk2
       
        #fp32
        qk2 = tl.dot(q.to(tl.float32), k2, allow_tf32=False)
        qk2 = tl.where(tl.arange(0, 16)[:, None]>= tl.arange(0,16)[None, :], qk2, 0.)
        tl.store(A_ptr + q_high * stride_a3 + q_high, qk2.to(A_ptr.dtype.element_ty))

        v2 = tl.load(V_ptr + q_high * stride_v3) 
        k_gv2 = tl.load(GV_ptr + q_high * stride_v3)
        k_gv2 = tl.exp(q_gv_normalizer[None, :] - k_gv2)
        v2 = v2 * k_gv2          

        #fp32
        output = tl.dot(qk2, v2, allow_tf32=False)        
        output = output * tl.load(Q_GV_ptr + q_high * stride_v3) 
    
        O_block_ptr = tl.make_block_ptr(
            base= O + v_offset,
            shape=(N_CTX, BLOCK_DMODEL_V),
            strides=(stride_v3, stride_v4),
            offsets=(start_m * BLOCK_N + q_high, 0),
            block_shape=(16, BLOCK_DMODEL_V),
            order=(1, 0)
        )
        prev = tl.load(O_block_ptr)
        output = output + prev
        tl.store(O_block_ptr, output.to(O.dtype.element_ty))




@triton.jit
def _bwd_kernel(Q, K, V, GK, GV, O, A, Q_GV, Q_GK,
                DO, 
                DQ, DK, DV, DGK, DGV,
                stride_q1, stride_q2, stride_q3, stride_q4,
                stride_v1, stride_v2, stride_v3, stride_v4,
                stride_a1, stride_a2, stride_a3, stride_a4,
                Z, H, N_CTX, D,
                BLOCK_DMODEL_QK: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr
                ):                    


    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    qk_offset = off_hz * stride_q2
    v_offset = off_hz * stride_v2
    a_offset = off_hz * stride_a2 

    lo = 0
    hi = BLOCK_N 

    # first compute dk and dv. then dqk is automically available for dq. so i don't need that much recomputation.
    # for memory consideration, i just put dA in-place in A. No need to allocate an additional memory at all without any conflicts. Keep an mind on it if you wanna read the code and do not be confused about what i am doing.
    Q_ptr = Q + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    Q_GK_ptr = Q_GK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    Q_GV_ptr = Q_GV + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    DO_ptr = DO + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    O_ptr = O + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3
    
    GK_Q_ptr = GK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    DGK_Q_ptr = DGK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    GV_ptr = GV + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    DGV_ptr = DGV + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    A_ptr = A + a_offset + (start_m * BLOCK_M) * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a3

    # prev-compute do, dgv (contributed by the q side)
    for q_high in range(lo, hi, 16):
        # precomputation of do. In-place store. 
        do = tl.load(DO_ptr + q_high * stride_v3)    
        o = tl.load(O_ptr + q_high * stride_v3)        
        tl.store(DGV_ptr + q_high * stride_v3, (do * o).to(DGV.dtype.element_ty))
        
        # q_gv_normalizer = tl.load().to(tl.float32)
        # # this can also be stored in the forward pass.
        # q_gv = tl.load(GV + v_offset + (start_m * BLOCK_N + q_high + tl.arange(0, 16)[:, None]) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :]).to(tl.float32)
        # q_gv = tl.exp(q_gv - q_gv_normalizer[None, :])

        q_gv = tl.load(Q_GV_ptr + q_high * stride_v3)
        do = do * q_gv.to(do.dtype)        
        tl.store(DO_ptr + q_high * stride_v3, do.to(DO.dtype.element_ty))                

        # precomputation of q'. Not in-place store cuz we need the origin copy of q later.
        # feel like we can save in the fwd pass. let's do it. 
        # DONE.
        # q = tl.load(Q_ptr + q_high * stride_q3) 
    
    tl.debug_barrier()
    # compute dk, dv, dv_gv (load, add then save), dk_dk (load, add then save)
    # tl.dot all in bf16
    K_ptr = K + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    DK_ptr = DK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    V_ptr = V + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    DV_ptr = DV + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    GK_K_ptr = GK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    DGK_K_ptr = DGK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    for k_high in range(lo, hi-16, 16):
        k = tl.load(K_ptr + k_high * stride_q3)
        k_gk = tl.load(GK_K_ptr + k_high * stride_q3)
        v = tl.load(V_ptr + k_high * stride_v3)
        k_gv = tl.load(GV_ptr + k_high * stride_v3)        
        dk = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)
        dgk = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)
        dv = tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32)
        dgv = tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32)

        for q_high in range(k_high + 16, hi, 16): 
            # already pre-compute
            do = tl.load(DO_ptr + q_high * stride_v3)
            qk = tl.load(A_ptr + q_high * stride_a3 + k_high)            
            # load normalizer.
            q_gv_normalizer = tl.load(GV + v_offset + (start_m * BLOCK_N + q_high) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)
            k_gv2 = tl.exp(q_gv_normalizer[None, :] - k_gv)
            v2 = v * k_gv2.to(v.dtype)                        
            dv2 = tl.dot(tl.trans(qk), do, allow_tf32=False)                        
            dv += dv2 * k_gv2
            dgv -= dv2 * v * k_gv2              

            dqk = tl.dot(do, tl.trans(v2), allow_tf32=False)                     
                        
            # dqk is then saved in-place in A because qk will not be used anymore. QK is only used when computing the gradient of v, dgv.
            tl.store(A_ptr + q_high * stride_a3 + k_high, dqk.to(A_ptr.dtype.element_ty))

            #compute k_gk
            q_gk_normalizer = tl.load(GK + qk_offset + (start_m * BLOCK_N + q_high) * stride_q3 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)
            k_gk2 = tl.exp(q_gk_normalizer[None, :] - k_gk)

            q = tl.load(Q_GK_ptr + q_high * stride_q3)                                                

            # compute dk 
            # bf16
            dk2 = tl.dot(tl.trans(dqk).to(q.dtype), q, allow_tf32=False)

            dk += dk2 * (k_gk)
            dgk -= dk2 * k * (k_gk)

  
        tl.store(DK_ptr +  k_high * stride_q3, (dk).to(DK.dtype.element_ty))
        
        DGK_ptr = DGK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3 + k_high * stride_q3
        prev = tl.load(DGK_ptr)
        tl.store(DGK_ptr, ((dgk) + prev).to(DGK.dtype.element_ty))

        tl.store(DV_ptr +k_high * stride_v3, (dv).to(DK.dtype.element_ty))
        
        # DGV_ptr = DGV + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3 + k_high * stride_v3
        prev = tl.load(DGV_ptr + k_high * stride_v3)
        tl.store(DGV_ptr + k_high * stride_v3, ((dgv) + prev).to(DGV.dtype.element_ty))


    tl.debug_barrier()

    # compute inter chunk dq. dq_gk. 
    for q_high in range(lo, hi, 16):

        # q_gk = tl.load(GK_Q_ptr + q_high * stride_q3).to(tl.float32)
        # q_gk = tl.exp(q_gk - q_normalizer[None, :])

        dq2 = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)
        q_normalizer = tl.load(GK + qk_offset + (start_m * BLOCK_N + q_high) * stride_q3 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)
        
        for k_high in range(0, q_high, 16):
            k = tl.load(K_ptr + k_high * stride_q3)
            k_gk = tl.load(GK_K_ptr + k_high * stride_q3)            
            
            k_gk = tl.exp(q_normalizer[None, :] - k_gk)
            k2 = k * k_gk.to(k.dtype)
            dqk = tl.load(A_ptr + q_high * stride_a3 + k_high)            
            dq2 += tl.dot(dqk.to(k2.dtype), k2, allow_tf32=False)

        q = tl.load(Q_ptr + q_high * stride_q3) 
        q_gk = tl.load(Q_GK_ptr + q_high * stride_q3)        
        dq = dq2 * q_gk 
        dq_gk = dq2 * q * q_gk 
        
        DQ_ptr = DQ + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3 + q_high * stride_q3
        tl.store(DQ_ptr, dq)

        DGK_Q_ptr = DGK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3 + q_high * stride_q3
        tl.store(DGK_Q_ptr, dq_gk)
    
    tl.debug_barrier()
    

    # compute inner chunk dq.
    for q_high in range(lo, hi, 16):

        q_gv_normalizer = tl.load(GV + v_offset + (start_m * BLOCK_N + q_high) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)
        q_normalizer = tl.load(GK + qk_offset + (start_m * BLOCK_N + q_high) * stride_q3 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)

        v = tl.load(V_ptr + q_high * stride_v3)
        k_gv = tl.load(GV_ptr + q_high * stride_v3)
        k_gv = tl.exp(q_gv_normalizer[None, :] - k_gv)
        v2 = v * k_gv     

        # already include q_gv, no worry here.
        do = tl.load(DO_ptr + q_high * stride_v3)

        qk = tl.load(A_ptr + q_high * stride_a3 + q_high)
        # qk = tl.where(tl.arange(0, 16)[:, None]>= tl.arange(0, 16)[None, :], qk, 0.).to(do.dtype)
        
        #bf16
        dv2 = tl.dot(tl.trans(qk), do, allow_tf32=False)        

        dv = dv2 * k_gv
        prev_dv = tl.load(DV_ptr + q_high * stride_v3)
        tl.store(DV_ptr + q_high * stride_v3, (dv + prev_dv).to(DV.dtype.element_ty))
        
        dgv = -dv2 * v * k_gv
        prev_dgv = tl.load(DGV_ptr + q_high * stride_v3)
        tl.store(DGV_ptr + q_high * stride_v3, (dgv + prev_dgv).to(DGV.dtype.element_ty))

        dqk = tl.dot(do.to(tl.float32), tl.trans(v2), allow_tf32=False)        
        k = tl.load(K_ptr + q_high * stride_q3)
        k_gk = tl.load(GK_K_ptr + q_high * stride_q3)
        k_gk = tl.exp(q_normalizer[None, :] - k_gk)
        k2 = k.to(tl.float32) * k_gk

        dq2 = tl.dot(dqk, (k2), allow_tf32=False)
        q = tl.load(Q_ptr + q_high * stride_q3)
        q_gk = tl.load(Q_GK_ptr + q_high * stride_q3)
        dq = dq2 * q_gk 
        dq_gk = dq2 * q * q_gk 

        DQ_ptr = DQ + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3 + q_high * stride_q3
        # prev = tl.load(DQ_ptr) + dq 
        tl.store(DQ_ptr, (dq).to(DQ.dtype.element_ty))

        # prev = tl.load(DGK_ptr + q_high * stride_q3)
        # tl.store(DGK_ptr, (prev + dq_gk).to(DGK.dtype.element_ty))                    
        

    

class FlashGRet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, gk, gv, BLOCK_N = 64):
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
    
        # assert gk.dtype == gv.dtype == torch.float32
        
        # for now.
        BLOCK_M = BLOCK_N
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        # right
        o = torch.zeros_like(v)

        grid = (triton.cdiv(q.shape[2], BLOCK_N), q.shape[0] * q.shape[1], 1)

        q_gv = torch.empty_like(gv)
        q_gk = torch.empty_like(gk)

        # batch, head, L, chunk_size.
        A = torch.empty(q.shape[0], q.shape[1], q.shape[2],  BLOCK_N, device=q.device, dtype=q.dtype)

        assert q.dtype == k.dtype == v.dtype  

        _fwd_kernel[grid](
            q, k, v, gk, gv, o, A, q_gv, q_gk,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],            
            
            BLOCK_N=BLOCK_N, BLOCK_DMODEL_QK=Lk, BLOCK_M=BLOCK_M, 
            BLOCK_DMODEL_V=Lv, num_warps=8, num_stages=4
        )

    

        ctx.save_for_backward(q, k, v, gk, gv, o, A, q_gv, q_gk)
        ctx.grid = grid
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_DMODEL_QK = Lk
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_DMODEL_V = Lv
        ctx.head = q.shape[1]
        return o

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        q, k, v, gk, gv, o, A, q_gv, q_gk = ctx.saved_tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dgk = torch.zeros_like(gk)
        dgv = torch.zeros_like(gv)

        BLOCK_N = ctx.BLOCK_N
        # for now.
        BLOCK_M = BLOCK_N
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

        _bwd_kernel[ctx.grid](
            q,  k, v, gk, gv, o,A, q_gv, q_gk,
            do, 
            dq, dk, dv, dgk, dgv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],            
            BLOCK_N=BLOCK_N, BLOCK_DMODEL_QK=Lk, BLOCK_M=BLOCK_M, 
            BLOCK_DMODEL_V=Lv, num_warps=16, num_stages=1
        )

        return dq.to(q), dk.to(k), dv.to(v), dgk.to(gk), dgv.to(gv), None

    

# def compute_inner_mixed(query, key, value, decay_key, decay_value, chunk_size=64):
#     intra = 


def compute_inner(query, key, value, decay_key, decay_value, chunk_size=64):
    query = rearrange(query, 'b h (n c) d -> b h n c d', c=chunk_size)
    key = rearrange(key, 'b h (n c) d -> b h n c d', c=chunk_size)
    value = rearrange(value, 'b h (n c) d -> b h n c d', c=chunk_size)

    mask = torch.triu(torch.ones(query.shape[-2], key.shape[-2]), diagonal=1).bool().to(query.device)

    original_dtype = query.dtype
    decay_key = decay_key.float().exp()
    decay_value = decay_value.float().exp()    
    
    query = query.float()
    key = key.float()
    value = value.float()

    query = (query * decay_key)
    key = key / decay_key 
    qk = (query @ key.transpose(-1, -2)).masked_fill_(mask, 0)     
    value = value / decay_value 
    return ((qk @ value) * decay_value).to(original_dtype)


if __name__ == "__main__":
    B = 32
    H = 4
    L = 2048
    D_QK = 256
    D_V = 512
    requires_grad = True


    dtype = torch.bfloat16
    q = (torch.rand(B, H, L, D_QK, device='cuda').to(dtype)).requires_grad_(requires_grad)  
    k = torch.rand(B, H, L, D_QK, device='cuda').to(dtype).requires_grad_(requires_grad)
    v = torch.rand(B, H, L, D_V, device='cuda').to(dtype).requires_grad_(requires_grad)
    gk = torch.randn(B, H, L, D_QK, device='cuda').to(dtype).requires_grad_(requires_grad)
    gv = torch.randn(B, H, L, D_V, device='cuda').to(dtype).requires_grad_(requires_grad)

    gk3 = F.logsigmoid(gk) / 16
    gv3 = F.logsigmoid(gv) / 16

    gk3 = gk3.clamp(min=-5)
    gv3 = gv3.clamp(min=-5)

    gk1 = (gk3).cumsum(-2) 
    gv1 = (gv3).cumsum(-2) 

    gk2 = (rearrange(gk3, 'b h (n c) d -> b h n c d', c=64)).cumsum(-2)
    gv2 = (rearrange(gv3, 'b h (n c) d -> b h n c d', c=64)).cumsum(-2)

    # o = FlashGRet.apply(q, k, v, gk1, gv1)
    # o.sum().backward(retain_graph=True)

    # target = [q, k, v, gk, gv]
    # grad1= []
    # grad2= []
    # for s in target:
    #     grad1.append(s.grad.clone())
    #     s.grad.zero_()
    
    # o2 = rearrange(compute_inner(q, k, v, gk2, gv2), 'b h n c d -> b h (n c) d')
    # o2.sum().backward(retain_graph=True)

    # for s in target:
    #     grad2.append(s.grad.clone())
    #     s.grad.zero_()


    # breakpoint()

    
    for _ in range(100):
        o = FlashGRet.apply(q, k, v, gk1, gv1)
        o2 = rearrange(compute_inner(q, k, v, gk2, gv2), 'b h n c d -> b h (n c) d')

    print("warm up done.")

    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        o = FlashGRet.apply(q, k, v, gk1, gv1)
        if requires_grad:
            o.sum().backward(retain_graph=True)    
            
    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}, fn:{FlashGRet.apply}")

    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        o2 = rearrange(compute_inner(q, k, v, gk2, gv2), 'b h n c d -> b h (n c) d')
        if requires_grad:
            o2.sum().backward(retain_graph=True)    
    
    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}, fn:{compute_inner}")



