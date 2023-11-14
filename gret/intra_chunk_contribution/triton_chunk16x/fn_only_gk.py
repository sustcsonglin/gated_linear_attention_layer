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
    
    K_ptr = K + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q4 
    
    GK_K_ptr = GK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q4 

    GK_Q_ptr = GK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 

    A_ptr = A + a_offset + (start_m) * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4 

    for q_high in range(16, hi, 16):

        q = tl.load(Q_ptr + q_high * stride_q4)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4).to(tl.float32)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + q_high * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)
        q_gk2 = tl.exp(q_gk - q_normalizer[None, :])
        q = q * q_gk2.to(q.dtype)

        #inter-chunk bf16
        for k_high in range(0, q_high, 16):
            k = tl.load(K_ptr + k_high * stride_q4)
            k_gk = tl.load(GK_K_ptr + k_high * stride_q4).to(tl.float32)            
            k_gk = tl.exp(q_normalizer[:, None] - k_gk)
            k = k * k_gk.to(k.dtype)
            qk = tl.dot(q, k, allow_tf32=False)            
            tl.store(A_ptr + q_high * stride_a4 + k_high, qk.to(A_ptr.dtype.element_ty))    


    ## intra chunk fp32
    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4).to(tl.float32)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + q_high * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)
        q_gk2 = tl.exp(q_gk - q_normalizer[None, :])
        q = q * q_gk2
        q_gk3 = tl.exp(q_normalizer[None, :] - q_gk)
        k = tl.load(K_ptr + q_high * stride_q4)
        k = k * tl.trans(q_gk3)

        qk = tl.dot(q, k, allow_tf32=False)
        qk = tl.where(tl.arange(0, 16)[:, None]>=tl.arange(0, 16)[None, :], qk, 0.)
        tl.store(A_ptr + q_high * stride_a4 + q_high, qk.to(A_ptr.dtype.element_ty))    




@triton.jit
def _bwd_kernel_dqk(Q, K, GK, DA,                
                DQ, 
                DK, DGK,
                stride_q1, stride_q2, stride_q3, stride_q4,
                stride_a1, stride_a2, stride_a3, stride_a4,
                Z, H, N_CTX, D,
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

    # DGK_Q_ptr = DGK + qk_offset + (start_m) * stride_q3+ tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    DA_ptr = DA + a_offset + (start_m) * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4 

    # inter chunk dq. bf16
    for q_high in range(lo+16, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4) 
        
        q_normalizer = tl.load(GK + qk_offset + (start_m * stride_q3)+ q_high * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)

        # q2 = q * q_gk.to(q.dtype)

        dq2 = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)

        for k_high in range(0, q_high, 16):
            k = tl.load(K_ptr + k_high * stride_q4)
            k_gk = tl.load(GK_K_ptr + k_high * stride_q4).to(tl.float32)            
            dqk = tl.load(DA_ptr + q_high * stride_a4 + k_high)
            k_gk = tl.exp(q_normalizer[None, :] - k_gk)
            k = k * k_gk.to(k.dtype)
            dq2 += tl.dot(dqk, k, allow_tf32=False)
        

        dq2 = dq2.to(q.dtype)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4).to(tl.float32)
        q_gk = tl.exp(q_gk - q_normalizer[None, :])
        dq = dq2 * q_gk.to(q.dtype) 
        dq_gk = dq * q
                
        DQ_ptr = DQ + qk_offset + (start_m) * stride_q3+ tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 + q_high * stride_q4
        tl.store(DQ_ptr, dq.to(DQ_ptr.dtype.element_ty))

        DGK_Q_ptr = DGK + qk_offset + (start_m) * stride_q3+ tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 + q_high * stride_q4
        # prev = tl.load(DGK_Q_ptr)
        tl.store(DGK_Q_ptr, dq_gk.to(DGK_Q_ptr.dtype.element_ty))

    tl.debug_barrier()


    
    for k_high in range(lo, hi-16, 16):
        k = tl.load(K_ptr + k_high * stride_q4)
        k_gk = tl.load(GK_K_ptr + k_high * stride_q4)
        dk = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)
        dgk = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)

        for q_high in range(k_high+16, hi, 16):
            q = tl.load(Q_ptr + q_high * stride_q4) 
            q_normalizer = tl.load(GK + qk_offset + (start_m * stride_q3)+ q_high * stride_q4 + tl.arange(0,
            BLOCK_DMODEL_QK)).to(tl.float32)
            q_gk = tl.load(GK_Q_ptr + q_high * stride_q4).to(tl.float32)
            q_gk = tl.exp(q_gk - q_normalizer[None, :]).to(q.dtype)
            q = q * q_gk
            dqk = tl.load(DA_ptr + q_high * stride_a4 + k_high)

            k_gk2 = tl.exp(q_normalizer[None, :] - k_gk)
            
            dk2 = tl.dot(tl.trans(dqk), q, allow_tf32=False)
            dk += dk2 * k_gk2
            dgk -= dk2 * k * k_gk2


        DK_ptr = DK + qk_offset + (start_m) * stride_q3+ tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 + k_high * stride_q4
        tl.store(DK_ptr, dk.to(DK_ptr.dtype.element_ty))

        DGK_K_ptr = DGK + qk_offset + (start_m) * stride_q3+ tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 + k_high * stride_q4
        prev = tl.load(DGK_K_ptr)
        tl.store(DGK_K_ptr,  (prev + dgk).to(DGK_K_ptr.dtype.element_ty))

    tl.debug_barrier()

    DK_ptr = DK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    DGK_K_ptr = DGK + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    DQ_ptr = DQ + qk_offset + (start_m) * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4

    ## intra chunk, fp32.
    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4).to(tl.float32)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + q_high * stride_q4 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)
        q_gk2 = tl.exp(q_gk - q_normalizer[None, :])
        q2 = q * q_gk2
        q_gk3 = tl.exp(q_normalizer[None, :] - q_gk)

        k = tl.load(K_ptr + q_high * stride_q4)
        k2 = k * q_gk3

        dqk = tl.load(DA_ptr + q_high * stride_a4 + q_high)
        dqk = tl.where(tl.arange(0, 16)[:, None]>=tl.arange(0, 16)[None, :], dqk, 0.)

        dk2 = tl.dot(tl.trans(dqk), q2, allow_tf32=False)        
        dk = dk2 * q_gk3
        prev_dk = tl.load(DK_ptr + q_high * stride_q4)
        tl.store(DK_ptr + q_high * stride_q4, (dk + prev_dk).to(DK_ptr.dtype.element_ty))

        dgk = - dk * k
        dq2 = tl.dot(dqk, k2, allow_tf32=False)
        dq = dq2 * q_gk2

        prev_dq = tl.load(DQ_ptr + q_high * stride_q4)
        tl.store(DQ_ptr + q_high * stride_q4, (dq + prev_dq).to(DQ_ptr.dtype.element_ty))

        dgk += dq * q
        prev_dq_gk = tl.load(DGK_K_ptr + q_high * stride_q4)
        tl.store(DGK_K_ptr + q_high * stride_q4, (dgk + prev_dq_gk).to(DGK_K_ptr.dtype.element_ty))
        






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



class FlashGRet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, gk):
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
            
        # assert q.dtype == k.dtype == v.dtype  

        _fwd_kernel_compute_A[grid](
            q, k, gk, A,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],            
            BLOCK_N=BLOCK_N, BLOCK_DMODEL_QK=Lk, BLOCK_M=BLOCK_M, num_warps=8, num_stages=8
        )

        ctx.save_for_backward(q, k, gk)
        ctx.grid = grid
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_DMODEL_QK = Lk
        ctx.BLOCK_N = BLOCK_N
        ctx.head = q.shape[1]
        return A 


    @staticmethod
    def backward(ctx, dA):
        dA = dA.contiguous()
        q, k, gk = ctx.saved_tensors

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dgk = torch.empty_like(gk)
         
        BLOCK_N = ctx.BLOCK_N
        # for now.
        BLOCK_M = BLOCK_N
        # shape constraints
        Lq, Lk = q.shape[-1], k.shape[-1]

        _bwd_kernel_dqk[ctx.grid](
            q, k, gk, dA,
            dq, 
            dk, dgk,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            dA.stride(0), dA.stride(1), dA.stride(2), dA.stride(3),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],
            BLOCK_N=BLOCK_N, BLOCK_DMODEL_QK=Lk, BLOCK_M=BLOCK_M, num_warps=16, num_stages=1
        )
        
        return dq, dk, dgk, None



    




if __name__ == "__main__":
    B = 32
    H = 8
    L = 2048
    D_QK = 256
    D_V = 256


    requires_grad = True
    chunk_size = 64
    num_chunk = L // chunk_size


    dtype = torch.float32
    q = (torch.rand(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype)).requires_grad_(requires_grad)  
    k = torch.rand(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype).requires_grad_(requires_grad)
    v = torch.rand(B, H, num_chunk, chunk_size, D_V, device='cuda').to(dtype).requires_grad_(requires_grad)
    gk = torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype).requires_grad_(requires_grad)

    # gv = torch.randn(B, H, L, D_V, device='cuda').requires_grad_(requires_grad)
    gk3 = F.logsigmoid(gk) / 8


    # gv3 = F.logsigmoid(gv) / 16
    # gk3 = gk3.clamp(min=-5)
    # gv3 = gv3.clamp(min=-5)
    # gk = (gk3).cumsum(-2)
    # gv1 = (gv3).cumsum(-2) 
    # gk2 = (rearrange(gk3, 'b h (n c) d -> b h n c d', c=64)).cumsum(-2)
    # gv2 = (rearrange(gv3, 'b h (n c) d -> b h n c d', c=32)).cumsum(-2)
    # breakpoint()
    print("starting.")



    o = FlashGRet.apply(q, k, gk3)
    o2 = compute_inner_A(q, k, gk3)
        
    for _ in range(200):
        o = FlashGRet.apply(q, k, gk) @ v 
        if requires_grad:
            o.sum().backward(retain_graph=True)    
        o2 = compute_inner(q, k, v, gk)
        if requires_grad:
            o2.sum().backward(retain_graph=True)    

        breakpoint()

    print("warm up done.")
    print(o-o2)


    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(1000):
        A = FlashGRet.apply(q, k, gk) 
        o = A @ v
        if requires_grad:
            o.sum().backward(retain_graph=True)    


    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}, fn:{FlashGRet.apply}")


    torch.cuda.synchronize()
    start = time.time()
    

    # for _ in range(1000):
    #     o2 = compute_inner(q, k, v, gk)
    #     if requires_grad:
    #         o2.sum().backward(retain_graph=True)    


    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}, fn:{compute_inner}")

    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(1000):
        o2 = (q @ k.transpose(-1,-2)) @ v
        if requires_grad:
            o2.sum().backward(retain_graph=True)    
    
    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}, fn:{compute_inner}")

    torch.cuda.synchronize()
    start = time.time()
    
    # for _ in range(1000):
    #     # q = rearrange(q, 'b h (n c) d -> b h n c d', c=64)
    #     # k = rearrange(k, 'b h (n c) d -> b h n c d', c=64)
    #     # v = rearrange(v, 'b h (n c) d -> b h n c d', c=64)
    #     # # gk = rearrange(gk, 'b h (n c) d -> b h n c d', c=64)
    #     o = cuda_compute_intra(q, k, gk) @ v
    #     if requires_grad:
    # #         o.sum().backward(retain_graph=True)

    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"scan gret onc, time:{end - start}, fn:{compute_inner}")    

