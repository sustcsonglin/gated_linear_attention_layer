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

@triton.jit
def _fwd_kernel_compute_A(
    Q, K, GK, 
    A, 
    stride_q1, stride_q2, stride_q3, stride_q4,
    stride_a1, stride_a2, stride_a3, stride_a4,
    Z, H, N_CTX, D,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr,
):
    # Q block的index
    start_m = tl.program_id(0)
    # Batch & Head 的index
    off_hz = tl.program_id(1)

    qk_offset = off_hz * stride_q2
    a_offset = off_hz * stride_a2
    
    # initialize offsets
    # initialize pointer to m and l    
    # load q: it will stay in SRAM throughout
    # loop over k, v and update accumulator

    lo = 0
    hi = BLOCK_N 

    # shape: BLOCK_DMODLE_QK
    # 对应的是每个Q Block第一个的

    Q_ptr = Q + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3
    
    K_ptr = K + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q3

    GK_K_ptr = GK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q3

    GK_Q_ptr = GK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    A_ptr = A + a_offset + (start_m * BLOCK_M) * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a3 + (start_m * BLOCK_M)

    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q3).to(tl.float32)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q3).to(tl.float32)
        
        q_normalizer = tl.load(GK + qk_offset + (start_m * BLOCK_N + q_high) * stride_q3 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)
        q_gk = tl.exp(q_gk - q_normalizer[None, :])
        q = q * q_gk

        for k_high in range(0, q_high + 16, 16):
            k = tl.load(K_ptr + k_high * stride_q3).to(tl.float32)
            k_gk = tl.load(GK_K_ptr + k_high * stride_q3).to(tl.float32)            
            k_gk = tl.exp(q_normalizer[:, None] - k_gk)
            k = k * k_gk
            #fp32            
            qk = tl.dot(q, k, allow_tf32=False)
            if k_high == q_high:
                qk = tl.where(tl.arange(0, 16)[:, None]>= tl.arange(0,16)[None, :], qk, 0)
            tl.store(A_ptr + q_high * stride_a3 + k_high * stride_a4, qk.to(A_ptr.dtype.element_ty))

@triton.jit
def _fwd_compute_O(
    A, V, GV, O, 
    stride_a1, stride_a2, stride_a3, stride_a4,
    stride_v1, stride_v2, stride_v3, stride_v4,
    Z, H, N_CTX, D,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr,
):
    # Q block的index
    start_m = tl.program_id(0)
    # Batch & Head 的index
    off_hz = tl.program_id(1)
    a_offset = off_hz * stride_a2
    v_offset = off_hz * stride_v2
    
    # initialize pointer to m and l
    # load q: it will stay in SRAM throughout
    # loop over k, v and update accumulator
    lo = 0
    hi = BLOCK_N 

    # shape: BLOCK_DMODLE_QK
    # 对应的是每个Q Block第一个的


    V_ptr = V + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3
    
    GV_ptr = GV + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    A_ptr = A + a_offset + (start_m * BLOCK_M) * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a3 + (start_m * BLOCK_M)

    for q_high in range(lo, hi, 16):

        q_gv_normalizer = tl.load(GV + v_offset + (start_m * BLOCK_N + q_high) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)

        acc = tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32)

        for k_high in range(0, q_high + 16, 16):
            # k = tl.load(K_ptr + k_high * stride_q3).to(tl.float32)
            # k_gk = tl.load(GK_K_ptr + k_high * stride_q3).to(tl.float32)            
            # k_gk = tl.exp(q_normalizer[:, None] - k_gk)
            # k = k * k_gk
            # #fp32            
            # qk = tl.dot(q, k, allow_tf32=False)
            # if k_high == q_high:
            #     qk = tl.where(tl.arange(0, 16)[:, None]>= tl.arange(0,16)[None, :], qk, 0)
            qk = tl.load(A_ptr + q_high * stride_a3 + k_high * stride_a4)
                    
            v = tl.load(V_ptr + k_high * stride_v3).to(tl.float32)
            k_gv = tl.load(GV_ptr + k_high * stride_v3).to(tl.float32)
            k_gv = tl.exp(q_gv_normalizer[None, :] - k_gv)
            v = v * k_gv            
            output = tl.dot(qk, v, allow_tf32=False)        
            acc += output

        q_gv = tl.load(GV + v_offset + (start_m * BLOCK_N + q_high + tl.arange(0, 16)[:, None]) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :]).to(tl.float32)
        q_gv = tl.exp(q_gv - q_gv_normalizer[None, :])
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



### TODO: 感觉应该把KV share来做，而不是反复的load and write. To reduce the IO cost.
@triton.jit
def _bwd_kernel_dav(V, GV, A, O, 
                DO, DA,
                DV, DGV, 
                stride_a1, stride_a2, stride_a3, stride_a4,
                stride_v1, stride_v2, stride_v3, stride_v4,
                Z, H, N_CTX, D,
                BLOCK_DMODEL_QK: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr
                ):
    
    # Q block的index
    start_m = tl.program_id(0)
    # Batch & Head 的index
    off_hz = tl.program_id(1)

    a_offset = off_hz * stride_a2
    v_offset = off_hz * stride_v2

    lo = 0
    hi = BLOCK_N 
    
    DO_ptr = DO + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    O_ptr = O + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3
    
    V_ptr = V + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    DV_ptr = DV + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    GV_ptr = GV + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    DGV_ptr = DGV + v_offset + (start_m * BLOCK_M) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v3

    A_ptr = A + a_offset + (start_m * BLOCK_M) * stride_a3 + tl.arange(0, 16)[:, None] + tl.arange(0, 16)[None, :] * stride_a3 + (start_m * BLOCK_M)

    DA_ptr = DA + a_offset + (start_m * BLOCK_M) * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a3 + (start_m * BLOCK_M)

    for q_high in range(lo, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v3)    
        o = tl.load(O_ptr + q_high * stride_v3)
        
        gv_pre = tl.load(DGV_ptr + q_high * stride_v3)
        tl.store(DGV_ptr + q_high * stride_v3, (gv_pre + do * o))

        q_gv_normalizer = tl.load(GV + v_offset + (start_m * BLOCK_N + q_high) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)

        q_gv = tl.load(GV + v_offset + (start_m * BLOCK_N + q_high + tl.arange(0, 16)[:, None]) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :]).to(tl.float32)
        q_gv = tl.exp(q_gv - q_gv_normalizer[None, :])
        do = do * q_gv

        for k_high in range(0, q_high + 16, 16):
            kq = tl.load(A_ptr + q_high * stride_a3 + k_high * stride_a4)

            v = tl.load(V_ptr + k_high * stride_v3).to(tl.float32)
            k_gv = tl.load(GV_ptr + k_high * stride_v3).to(tl.float32)
            k_gv = tl.exp(q_gv_normalizer[None, :] - k_gv)
            v2 = v * k_gv            

            dv2 = tl.dot(kq, do, allow_tf32=False)
            dqk = tl.dot(do, tl.trans(v2), allow_tf32=False)

            if k_high == q_high:
                dqk = tl.where(tl.arange(0, 16)[:, None]>= tl.arange(0,16)[None, :], dqk, 0)
            
            tl.store(DA_ptr + q_high * stride_a3 + k_high * stride_a4, dqk)
 
            # update DV
            dv = dv2 * k_gv     
            dv_prev = tl.load(DV_ptr + k_high * stride_v3)
            tl.store(DV_ptr + k_high * stride_v3, dv + dv_prev)

            # update GV
            dkgv2 = -(dv2 * v * k_gv)
            dgv_prev = tl.load(DGV_ptr + k_high * stride_v3)
            tl.store(DGV_ptr + k_high * stride_v3, dkgv2 + dgv_prev)
                
    
## Most gradient should be fp32 when reloading and summation is needed.
## what gradient needn't?  My guess is DQ
@triton.jit
def _bwd_kernel_dqk(Q, K, GK, DA,                
                DQ, DK, DGK,
                stride_q1, stride_q2, stride_q3, stride_q4,
                stride_a1, stride_a2, stride_a3, stride_a4,
                Z, H, N_CTX, D,
                BLOCK_DMODEL_QK: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr
                ):

    # Q block的index
    start_m = tl.program_id(0)
    # Batch & Head 的index
    off_hz = tl.program_id(1)
    qk_offset = off_hz * stride_q2
    a_offset = off_hz * stride_a2

    lo = 0
    hi = BLOCK_N 

    # shape: BLOCK_DMODLE_QK
    # 对应的是每个Q Block第一个的

    Q_ptr = Q + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    DQ_ptr = DQ + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3
    
    K_ptr = K + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q3

    DK_ptr = DK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q3

    GK_K_ptr = GK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q3

    DGK_K_ptr = DGK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q3

    GK_Q_ptr = GK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    DGK_Q_ptr = DGK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3

    DA_ptr = DA + a_offset + (start_m * BLOCK_M) * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a3 + (start_m * BLOCK_M)

    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q3) 
        
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q3).to(tl.float32)
        q_normalizer = tl.load(GK + qk_offset + (start_m * BLOCK_N + q_high) * stride_q3 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)
        q_gk = tl.exp(q_gk - q_normalizer[None, :])
        q2 = q * q_gk

        dq2 = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)

        for k_high in range(0, q_high + 16, 16):
            k = tl.load(K_ptr + k_high * stride_q3).to(tl.float32)
            k_gk = tl.load(GK_K_ptr + k_high * stride_q3).to(tl.float32)            
            k_gk = tl.exp(q_normalizer[:, None] - k_gk)
            k2 = k * k_gk
    
            dqk = tl.load(DA_ptr + q_high * stride_a3 + k_high * stride_a4)
 
            # update dQ
            dq2 += tl.dot(dqk, tl.trans(k2), allow_tf32=False)

            # update dk
            dk2 = tl.dot(tl.trans(q2), dqk, allow_tf32=False)
            dk = dk2 * k_gk

            #update DK
            dk_prev = tl.load(DK_ptr + k_high * stride_q3)
            tl.store(DK_ptr + k_high * stride_q3, dk + dk_prev)
            
            #update GK
            dk_gk2 = -(dk2 * k * k_gk)
            dkgk_prev = tl.load(DGK_K_ptr + k_high * stride_q3)
            tl.store(DGK_K_ptr + k_high * stride_q3, dk_gk2 + dkgk_prev)
            
        dq = dq2 * q_gk 
        dq_gk = dq2 * q * q_gk 
        
        DQ_ptr = DQ + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3 + q_high * stride_q3
        tl.store(DQ_ptr, dq)

        DGK_Q_ptr = DGK + qk_offset + (start_m * BLOCK_M) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q3 + q_high * stride_q3
        prev = tl.load(DGK_Q_ptr)
        tl.store(DGK_Q_ptr, prev + dq_gk)
    

class FlashGRet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, gk, gv, BLOCK_N = 32):
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
    
        assert gk.dtype == gv.dtype == torch.float32
        
        # for now.
        BLOCK_M = BLOCK_N
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        # right
        o = torch.zeros_like(v)

        A = torch.empty(q.shape[0], q.shape[1], q.shape[2], q.shape[2], device=q.device)
        
        grid = (triton.cdiv(q.shape[2], BLOCK_N), q.shape[0] * q.shape[1], 1)

        assert q.dtype == k.dtype == v.dtype  

        _fwd_kernel_compute_A[grid](
            q, k,  gk, A,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],            
            BLOCK_N=BLOCK_N, BLOCK_DMODEL_QK=Lk, BLOCK_M=BLOCK_M, 
            BLOCK_DMODEL_V=Lv,
        )
        _fwd_compute_O[grid](A, v, gv, o,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],
            BLOCK_N=BLOCK_N, BLOCK_DMODEL_QK=Lk, BLOCK_M=BLOCK_M, 
            BLOCK_DMODEL_V=Lv
        )
    

        ctx.save_for_backward(q, k, v, gk, gv, A, o)
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
        q, k, v, gk, gv, A, o = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        dgk = torch.zeros_like(gk, dtype=torch.float32)
        dgv = torch.zeros_like(gv, dtype=torch.float32)
        dA = torch.zeros_like(A, dtype=torch.float32)

        
        BLOCK_N = ctx.BLOCK_N
        # for now.
        BLOCK_M = BLOCK_N
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

        _bwd_kernel_dav[ctx.grid](
            v, gv, A, o,
            do, dA,
            dv, dgv,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],            
            BLOCK_N=BLOCK_N, BLOCK_DMODEL_QK=Lk, BLOCK_M=BLOCK_M, 
            BLOCK_DMODEL_V=Lv,
        )
        _bwd_kernel_dqk[ctx.grid](
            q, k, gk, dA,
            dq, dk, dgk,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],
            BLOCK_N=BLOCK_N, BLOCK_DMODEL_QK=Lk, BLOCK_M=BLOCK_M,
            BLOCK_DMODEL_V=Lv
        )
        return dq.to(q), dk.to(k), dv.to(v), dgk.to(gk), dgv.to(gv), None
    



# def compute_inner_mixed(query, key, value, decay_key, decay_value, chunk_size=64):
#     intra = 


def compute_inner(query, key, value, decay_key, decay_value, chunk_size=32):
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
    B = 4
    H = 4
    L = 2048
    D_QK = 128
    D_V = 256
    requires_grad = True

    dtype = torch.bfloat16
    q = (torch.rand(B, H, L, D_QK, device='cuda').to(dtype)).requires_grad_(requires_grad)  
    k = torch.rand(B, H, L, D_QK, device='cuda').to(dtype).requires_grad_(requires_grad)
    v = torch.rand(B, H, L, D_V, device='cuda').to(dtype).requires_grad_(requires_grad)
    gk = torch.randn(B, H, L, D_QK, device='cuda') .requires_grad_(requires_grad)
    gv = torch.randn(B, H, L, D_V, device='cuda').requires_grad_(requires_grad)

    gk3 = F.logsigmoid(gk) / 16
    gv3 = F.logsigmoid(gv) / 16

    gk3 = gk3.clamp(min=-5)
    gv3 = gv3.clamp(min=-5)

    gk1 = (gk3).cumsum(-2) 
    gv1 = (gv3).cumsum(-2) 

    gk2 = (rearrange(gk3, 'b h (n c) d -> b h n c d', c=32)).cumsum(-2)
    gv2 = (rearrange(gv3, 'b h (n c) d -> b h n c d', c=32)).cumsum(-2)


    o = FlashGRet.apply(q, k, v, gk1, gv1)
    o.sum().backward(retain_graph=True)

    target = [q, k, v, gk, gv]

    grad1= []
    grad2= []
    for s in target:
        grad1.append(s.grad.clone())
        s.grad.zero_()
    
    o2 = rearrange(compute_inner(q, k, v, gk2, gv2), 'b h n c d -> b h (n c) d')
    o2.sum().backward(retain_graph=True)

    for s in target:
        grad2.append(s.grad.clone())
        s.grad.zero_()


    # breakpoint()

    
    for _ in range(100):
        o = FlashGRet.apply(q, k, v, gk1, gv1)
        o2 = rearrange(compute_inner(q, k, v, gk2, gv2), 'b h n c d -> b h (n c) d')

    print("warm up done.")
    print(o-o2)



    torch.cuda.synchronize()
    start = time.time()
    

    for _ in range(1000):
        o = FlashGRet.apply(q, k, v, gk1, gv1,32)
        if requires_grad:
            o.sum().backward(retain_graph=True)    
    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}, fn:{FlashGRet.apply}")

    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(1000):
        o2 = rearrange(compute_inner(q, k, v, gk2, gv2), 'b h n c d -> b h (n c) d')
        if requires_grad:
            o2.sum().backward(retain_graph=True)    
    
    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}, fn:{compute_inner}")


    torch.cuda.synchronize()
    start = time.time()
    

    for _ in range(1000):
        o2 = (q @ k.transpose(-1,-2))@ v
        if requires_grad:
            o2.sum().backward(retain_graph=True)    
    
    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}, fn:{compute_inner}")


