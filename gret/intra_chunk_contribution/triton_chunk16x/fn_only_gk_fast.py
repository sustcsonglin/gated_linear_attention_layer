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
from fn_only_gk import FlashGRet

@triton.jit
def _fwd_kernel_compute_A(
    Q, K, GK, 
    A, 
    stride_q1, stride_q2, stride_q3, stride_q4,
    stride_a1, stride_a2, stride_a3, stride_a4,
    Z, H, N_CTX, D,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_d = tl.program_id(2)

    qk_offset = off_hz * stride_q2 
    a_offset = off_hz * stride_a2 
    
    lo = 0
    hi = BLOCK_N 

    Q_ptr = Q + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    
    GK_Q_ptr = GK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 

    A_ptr = A + a_offset + (start_m) * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4 

    K_ptr = K + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 
    GK_K_ptr = GK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 

    # pre-process q and k.
    for k_high in range(lo, hi, 16):
        k = tl.load(K_ptr + k_high * stride_q4)
        k_gk = tl.load(GK_K_ptr + k_high * stride_q4).to(tl.float32)            
        k_gk_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + (k_high + 15) * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)        
        k = k * tl.exp(k_gk_normalizer[None, :] - k_gk)
        tl.store(K_ptr + k_high * stride_q4, k.to(K_ptr.dtype.element_ty))

        q = tl.load(Q_ptr + k_high * stride_q4)
        q_gk_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + (k_high) * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)        
        q = q * tl.exp(k_gk - q_gk_normalizer[None, :])
        tl.store(Q_ptr + k_high * stride_q4, q.to(Q_ptr.dtype.element_ty))

    tl.debug_barrier()
    
    # transpose
    K_ptr = K + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q4 
    GK_K_ptr = GK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q4 

    for q_high in range(16, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + q_high * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)

        #inter-chunk bf16
        for k_high in range(0, q_high, 16):
            k = tl.load(K_ptr + k_high * stride_q4)

            k_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + (k_high + 15) * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK))
            k = k * tl.exp(q_normalizer - k_normalizer)[:, None].to(k.dtype)

            qk = tl.dot(q, k, allow_tf32=False)            
            tl.store(A_ptr + q_high * stride_a4 + k_high, qk.to(A_ptr.dtype.element_ty))    

    tl.debug_barrier()

    ## intra chunk fp32
    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + q_high * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)
        k_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + (q_high + 15) * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK))

        k = tl.load(K_ptr + q_high * stride_q4)
        k = k * tl.exp(q_normalizer-k_normalizer)[:, None]

        #fp32
        qk = tl.dot(q.to(k.dtype), k, allow_tf32=False)
        qk = tl.where(tl.arange(0, 16)[:, None]>=tl.arange(0, 16)[None, :], qk, 0.)
        tl.store(A_ptr + q_high * stride_a4 + q_high, qk.to(A_ptr.dtype.element_ty))    

        




@triton.jit
def _bwd_kernel_dqk(Q, Q_origin, K, K_origin, GK, DA,                
                DQ, 
                DK, DGK,
                stride_q1, stride_q2, stride_q3, stride_q4,
                stride_a1, stride_a2, stride_a3, stride_a4,
                Z, H, N_CTX, D,
                NUM_SPLIT_QK: tl.constexpr,
                BLOCK_DMODEL_QK: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
                ):

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qk_offset = off_hz * stride_q2
    a_offset = off_hz * stride_a2

    lo = 0
    hi = BLOCK_N 

    Q_ptr = Q + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    DQ_ptr = DQ + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    
    K_ptr = K + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    GK_K_ptr = GK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    GK_Q_ptr = GK + qk_offset + (start_m) * stride_q3+ tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    DA_ptr = DA + a_offset + (start_m) * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4 

    # inter chunk dq. bf16
    for q_high in range(lo+16, hi, 16):        
        q_normalizer = tl.load(GK + qk_offset + (start_m * stride_q3)+ q_high * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)

        dq2 = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)

        for k_high in range(0, q_high, 16):
            k = tl.load(K_ptr + k_high * stride_q4)
            
            k_normalizer = tl.load(GK + qk_offset + (start_m * stride_q3)+ (k_high + 15) * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)
            dqk = tl.load(DA_ptr + q_high * stride_a4 + k_high)

            k = k * tl.exp(q_normalizer - k_normalizer)[None, :].to(k.dtype)
            dq2 += tl.dot(dqk, k, allow_tf32=False)
                
        DQ_ptr = DQ + qk_offset + (start_m) * stride_q3+ tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 + q_high * stride_q4
        tl.store(DQ_ptr, dq2.to(DQ_ptr.dtype.element_ty))


    tl.debug_barrier()


    DA_ptr = DA + a_offset + (start_m) * stride_a3 + tl.arange(0, 16)[:, None] + tl.arange(0, 16)[None, :] * stride_a4 

    for k_high in range(lo, hi-16, 16):
        dk = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)
        k_normalizer = tl.load(GK + qk_offset + (start_m * stride_q3)+ (k_high + 15) * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)          

        for q_high in range(k_high+16, hi, 16):
            q = tl.load(Q_ptr + q_high * stride_q4) 
            q_normalizer = tl.load(GK + qk_offset + (start_m * stride_q3)+ q_high * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)          
            dqk = tl.load(DA_ptr + q_high * stride_a4 + k_high)
            dk2 = tl.dot(dqk, q, allow_tf32=False)
            dk += dk2 * tl.exp(q_normalizer - k_normalizer)[None, :]

        DK_ptr = DK + qk_offset + (start_m) * stride_q3+ tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 + k_high * stride_q4
        tl.store(DK_ptr, dk.to(DK_ptr.dtype.element_ty))

    tl.debug_barrier()

    DK_ptr = DK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    DGK_K_ptr = DGK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    DQ_ptr = DQ + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    
    Q_origin_ptr = Q_origin + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    K_origin_ptr = K_origin + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    
    DA_ptr = DA + a_offset + (start_m) * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4 

    ## intra chunk, fp32.
    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + q_high * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)
        k_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + (q_high + 15) * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)
        k = tl.load(K_ptr + q_high * stride_q4)
        dqk = tl.load(DA_ptr + q_high * stride_a4 + q_high)
        dqk = tl.where(tl.arange(0, 16)[:, None]>=tl.arange(0, 16)[None, :], dqk, 0.)
        dk = tl.dot(tl.trans(dqk), q.to(dqk.dtype), allow_tf32=False)        
        dk = dk * tl.exp(q_normalizer - k_normalizer)[None, :]                
        prev_dk = tl.load(DK_ptr + q_high * stride_q4)
        dk += prev_dk


        gk = tl.load(GK_Q_ptr + q_high * stride_q4)
        gk2 = tl.exp(k_normalizer[None, :] - gk)        
        dk = dk * gk2

        tl.store(DK_ptr + q_high * stride_q4, dk.to(DK_ptr.dtype.element_ty))
        k_origin = tl.load(K_origin_ptr + q_high * stride_q4)
        dgk = -dk * k_origin 

        k = k * tl.exp(q_normalizer - k_normalizer)[None, :]

        dq = tl.dot(dqk, k, allow_tf32=False)    
        prev_dq = tl.load(DQ_ptr + q_high * stride_q4)
        dq += prev_dq
        gk = tl.exp(gk - q_normalizer[None, :])                            
        dq = dq * gk.to(dq.dtype)
        tl.store(DQ_ptr + q_high * stride_q4, dq.to(DQ_ptr.dtype.element_ty))

        q_origin = tl.load(Q_origin_ptr + q_high * stride_q4)
        dgk += dq * q_origin
        tl.store(DGK_K_ptr + q_high * stride_q4, (dgk).to(DGK_K_ptr.dtype.element_ty))
        







def compute_inner(query, key, value, decay_key):
    # query = rearrange(query, 'b h (n c) d -> b h n c d', c=chunk_size)
    # key = rearrange(key, 'b h (n c) d -> b h n c d', c=chunk_size)
    # value = rearrange(value, 'b h (n c) d -> b h n c d', c=chunk_size)

    mask = torch.triu(torch.ones(query.shape[-2], key.shape[-2]), diagonal=1).bool().to(query.device)
    original_dtype = query.dtype
    decay_key = decay_key.float().exp()
    # decay_value = decay_value.float().exp()    
    
    query = query.float()
    key = key.float()
    value = value.float()

    query = (query * decay_key)
    key = key / decay_key 
    qk = (query @ key.transpose(-1, -2)).masked_fill_(mask, 0)     
    return qk  @ value


def compute_inner_A(query, key, decay_key):
    # query = rearrange(query, 'b h (n c) d -> b h n c d', c=chunk_size)
    # key = rearrange(key, 'b h (n c) d -> b h n c d', c=chunk_size)
    # value = rearrange(value, 'b h (n c) d -> b h n c d', c=chunk_size)

    mask = torch.triu(torch.ones(query.shape[-2], key.shape[-2]), diagonal=1).bool().to(query.device)
    original_dtype = query.dtype
    decay_key = decay_key.float().exp()
    # decay_value = decay_value.float().exp()    
    
    query = query.float()
    key = key.float()        
    # value = value.float()

    query = (query * decay_key)
    key = key / decay_key 
    qk = (query @ key.transpose(-1, -2)).masked_fill_(mask, 0)     
    return qk  



class FlashGRet_Faster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, gk):
        q = q.contiguous()
        k = k.contiguous()
        gk = gk.contiguous()

        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")

        # assert gk.dtype == gv.dtype == torch.float32        
        # for now.
        BLOCK_M = BLOCK_N = q.shape[-2]

        # shape constraints
        Lq, Lk = q.shape[-1], k.shape[-1]

        A = torch.zeros(q.shape[0], q.shape[1], q.shape[2], BLOCK_N, BLOCK_N, device=q.device, dtype=q.dtype)        
        
        grid = (q.shape[2] , q.shape[0] * q.shape[1], 1)     

        q_exp = q.clone()
        k_exp = k.clone()

        # assert q.dtype == k.dtype == v.dtype                  
        _fwd_kernel_compute_A[grid](
            q_exp, k_exp, gk, A,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],            
            BLOCK_N=BLOCK_N, BLOCK_DMODEL_QK=Lk, BLOCK_M=BLOCK_M, num_warps=8, num_stages=8
        )

        ctx.save_for_backward(q, k, q_exp, k_exp, gk)
        ctx.grid = grid
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_DMODEL_QK = Lk
        ctx.BLOCK_N = BLOCK_N
        ctx.head = q.shape[1]
        return A 


    @staticmethod
    def backward(ctx, dA):
        dA = dA.contiguous()
        q, k, q_exp, k_exp, gk = ctx.saved_tensors

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dgk = torch.empty_like(gk)
         
        BLOCK_N = ctx.BLOCK_N
        # for now.
        BLOCK_M = BLOCK_N
        # shape constraints
        Lq, Lk = q.shape[-1], k.shape[-1]

        _bwd_kernel_dqk[ctx.grid](
            q_exp, q, k_exp, k, gk, dA,
            dq, 
            dk, dgk,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            dA.stride(0), dA.stride(1), dA.stride(2), dA.stride(3),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],
            BLOCK_N=BLOCK_N, BLOCK_DMODEL_QK=Lk, BLOCK_M=BLOCK_M, num_warps=8, num_stages=1
        )
        
        return dq, dk, dgk, None




    

if __name__ == "__main__":
    B = 32
    H = 4
    L = 2048
    D_QK = 256
    D_V = 512
    print("Hello.")

    requires_grad = False 
    chunk_size = 64
    num_chunk = L // chunk_size

    dtype = torch.bfloat16
    q = torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype).requires_grad_(requires_grad) 
    k = torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype).requires_grad_(requires_grad)
    
    gk = torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype).requires_grad_(requires_grad)

    # gv = torch.randn(B, H, L, D_V, device='cuda').requires_grad_(requires_grad)
    gk3 = F.logsigmoid(gk) / 16
    gk3 = gk3.cumsum(-2)
    # gk3 = gk3 * 0

    # o = FlashGRet_Faster.apply(q, k, gk3)
    
    # #gradient check

    # o.sum().backward(retain_graph=True)

    # target = [q, k, gk]

    # grad1= []
    # grad2= []
    # for s in target:
    #     grad1.append(s.grad.clone())
    #     s.grad.zero_()
    
    # o2 = FlashGRet.apply(q, k, gk3)    
    # o2.sum().backward(retain_graph=True)

    # for s in target:
    #     grad2.append(s.grad.clone())
    #     s.grad.zero_()
    
    # print( (o - o2).abs().max())
    

    # for a, b in zip(grad1, grad2):
        # print( (a  - b).abs().max())


    for _ in range(200):
        o =  FlashGRet.apply(q, k, gk3)    
        if requires_grad:
            o.sum().backward(retain_graph=True)    
        o2 =  FlashGRet_Faster.apply(q, k, gk3)    
        if requires_grad:
            o2.sum().backward(retain_graph=True)    

    print("warm up done.")
    print(o-o2)

    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(500):
        o =  FlashGRet.apply(q, k, gk3)    
        if requires_grad:
            o.sum().backward(retain_graph=True)    

    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}")

    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(500):
        o2 = FlashGRet_Faster.apply(q, k, gk3)    
        if requires_grad:
            o2.sum().backward(retain_graph=True)    

    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}")

    torch.cuda.synchronize()
    start = time.time()
    
    
