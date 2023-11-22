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

# from fn_only_gv import FlashGRet_O

@triton.jit
def _fwd_compute_O(
    A, V, GV, O, 
    stride_a1, stride_a2, stride_a3, stride_a4,
    stride_v1, stride_v2, stride_v3, stride_v4,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    a_offset = off_hz * stride_a2
    v_offset = off_hz * stride_v2
    off_v = tl.program_id(2)

    lo = 0
    hi = BLOCK_N 

    V_ptr = V + v_offset + (start_m) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4 + off_v * BLOCK_DMODEL_V

    O_ptr = O + v_offset + (start_m) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4 + off_v * BLOCK_DMODEL_V

    GV_ptr = GV + v_offset + (start_m) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4 + off_v * BLOCK_DMODEL_V

    A_ptr = A + a_offset + (start_m) * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4 

    # preprocess v.
    for k_high in range(lo, hi, 16):
        gv = tl.load(GV_ptr + k_high * stride_v4)
        v = tl.load(V_ptr + k_high * stride_v4)
        gv_normalizer = tl.load(GV + v_offset + (start_m) * stride_v3 + (k_high + 15) * stride_v4 + tl.arange(0, BLOCK_DMODEL_V) + off_v * BLOCK_DMODEL_V).to(tl.float32)
        v2 = v * tl.exp(gv_normalizer[None, :] - gv)
        tl.store(V_ptr + k_high * stride_v4, v2.to(V.dtype.element_ty))        

    tl.debug_barrier()
    
    for q_high in range(lo+16, hi, 16):
        q_gv_normalizer = tl.load(GV + v_offset + (start_m) * stride_v3 + q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V) + off_v * BLOCK_DMODEL_V).to(tl.float32)

        acc = tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32)        

        for k_high in range(0, q_high, 16):            
            qk = tl.load(A_ptr + q_high * stride_a4 + k_high)                    
            v = tl.load(V_ptr + k_high * stride_v4)
            k_gv_normalizer = tl.load(GV + v_offset + (start_m) * stride_v3 + (k_high + 15) * stride_v4 + tl.arange(0, BLOCK_DMODEL_V) + off_v * BLOCK_DMODEL_V).to(tl.float32)            

            #bf16
            output = tl.dot(qk.to(v.dtype), v, allow_tf32=False) * tl.exp(q_gv_normalizer - k_gv_normalizer).to(v.dtype)[None, :]     
            acc += output
        
        q_gv = tl.load(GV_ptr + q_high * stride_v4)
        q_gv = tl.exp(q_gv - q_gv_normalizer[None, :])
        acc *= q_gv


        tl.store(O_ptr + q_high * stride_v4, acc.to(O.dtype.element_ty))    
    
    tl.store(O_ptr, tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32).to(O_ptr.dtype.element_ty))
    tl.debug_barrier()

    for q_high in range(lo, hi, 16):

        qk = tl.load(A_ptr + q_high * stride_a4 + q_high)                            
        v = tl.load(V_ptr + q_high * stride_v4)
        
        k_gv_normalizer = tl.load(GV + v_offset + (start_m) * stride_v3 + (q_high + 15) * stride_v4 + tl.arange(0, BLOCK_DMODEL_V) + off_v * BLOCK_DMODEL_V).to(tl.float32)

        output = tl.dot(qk, v, allow_tf32=False)
        q_gv = tl.load(GV_ptr + q_high * stride_v4)        
        q_gv = tl.exp(q_gv-k_gv_normalizer[None, :])
        output = output * q_gv

        prev = tl.load(O_ptr + q_high * stride_v4)
        output += prev 
        tl.store(O_ptr + q_high * stride_v4, output.to(O.dtype.element_ty))


@triton.jit
def _bwd_kernel_dav(V, V_origin, GV, A, O, 
                DO, DA,
                DV, DGV, 
                Z, H, 
                stride_a1, stride_a2, stride_a3, stride_a4,
                stride_v1, stride_v2, stride_v3, stride_v4, 
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr
                ):
    
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_v = tl.program_id(2)

    a_offset = off_hz * stride_a2 + (start_m) * stride_a3
    da_offset = (off_v * Z * H + off_hz) * stride_a2 + start_m * stride_a3 
    v_offset = off_hz * stride_v2 + off_v * BLOCK_DMODEL_V + (start_m) * stride_v3

    lo = 0
    hi = BLOCK_N 

    DO_ptr = DO + v_offset + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4 

    O_ptr = O + v_offset  + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4 
    

    DV_ptr = DV + v_offset + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4

    GV_ptr = GV + v_offset + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4

    DGV_ptr = DGV + v_offset + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4

    A_ptr = A + a_offset + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4

    DA_ptr = DA + da_offset + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4 

    # pre-compute do*q_gv. in-place update
    # no synchronization need.
    for q_high in range(lo, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v4)    
        o = tl.load(O_ptr + q_high * stride_v4)
        tl.store(DGV_ptr + q_high * stride_v4, (do * o))                
        q_gv_normalizer = tl.load(GV + v_offset +  tl.arange(0, BLOCK_DMODEL_V) + q_high * stride_v4).to(tl.float32)
        q_gv = tl.load(GV_ptr + q_high * stride_v4)
        q_gv = tl.exp(q_gv - q_gv_normalizer[None, :])
        do = do * q_gv        
        tl.store(DO_ptr + q_high * stride_v4, do)
        
    tl.debug_barrier()

    V_ptr = V + v_offset + tl.arange(0, BLOCK_DMODEL_V)[:, None] + tl.arange(0, 16)[None, :] * stride_v4 

    for q_high in range(lo+16, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v4)            
        q_gv_normalizer = tl.load(GV + v_offset  + q_high * stride_v4 + tl.arange(0, 
        BLOCK_DMODEL_V)).to(tl.float32)
        
        for k_high in range(0, q_high, 16):
            v = tl.load(V_ptr + k_high * stride_v4)
            # k_gv = tl.load(GV_ptr + k_high * stride_v4)
            # k_gv = tl.exp(q_gv_normalizer[:, None] - k_gv)
            # bf16
            k_gv_normalizer = tl.load(GV + v_offset + (k_high + 15) * stride_v4 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)
            k_gv = tl.exp(q_gv_normalizer - k_gv_normalizer)
            v2 = v * k_gv.to(v.dtype)[:, None]                        
            dqk = tl.dot(do, v2, allow_tf32=False)                        
            # need synchronization
            tl.store(DA_ptr + q_high * stride_a4 + k_high, dqk.to(DA.dtype.element_ty))          
    
    tl.debug_barrier()

    A_ptr = A + a_offset + tl.arange(0, 16)[:, None] + tl.arange(0, 16)[ None, :] * stride_a4
    V_ptr = V + v_offset + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[ :, None] * stride_v4
    GV_ptr = GV + v_offset + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[ :, None] * stride_v4

    for k_high in range(0, hi, 16):        
        dv = tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32)
        k_gv = tl.load(GV_ptr + k_high * stride_v4)

        for q_high in range(k_high + 16, BLOCK_N, 16):
            do = tl.load(DO_ptr + q_high * stride_v4)                        
            kq = tl.load(A_ptr + q_high * stride_a4 + k_high)                    
            dv2 = tl.dot(kq, do, allow_tf32=False)            
            q_gv_normalizer = tl.load(GV + v_offset + q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)
            k_gv_normalizer = tl.load(GV + v_offset + (k_high + 15) * stride_v4 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)
            dv += dv2 * tl.exp(q_gv_normalizer - k_gv_normalizer)[None, :]

        tl.store(DV_ptr + k_high * stride_v4, dv.to(DV_ptr.dtype.element_ty))

    tl.debug_barrier()

    A_ptr = A + a_offset + tl.arange(0, 16)[:, None] + tl.arange(0, 16)[ None, :] * stride_a4 
    V_origin_ptr = V_origin + v_offset + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4 
    
    #intra-chunk
    for q_high in range(lo, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v4)            

        q_gv_normalizer = tl.load(GV + v_offset + q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)

        v = tl.load(V_ptr + q_high * stride_v4)

        # last chunk?
        k_gv_normalizer = tl.load(GV + v_offset + (q_high + 15) * stride_v4 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)

        # k_gv = tl.load(GV_ptr + q_high * stride_v4)
        k_gv =  tl.exp(q_gv_normalizer - k_gv_normalizer)[None, :] 

        v2 = v * k_gv.to(v.dtype)

        dqk = tl.dot(do.to(v2.dtype), tl.trans(v2), allow_tf32=False)
        tl.store(DA_ptr + q_high * stride_a4 + q_high, dqk.to(DA_ptr.dtype.element_ty))

        kq = tl.load(A_ptr + q_high * stride_a4 + q_high)

        dv2 = tl.dot(kq, do, allow_tf32=False)
        
        dv = dv2 * k_gv

        prev_dv = tl.load(DV_ptr + q_high * stride_v4)
        dv += prev_dv
        # tl.store(DV_ptr + q_high * stride_v4, (prev_dv + dv).to(DV.dtype.element_ty))

        #  = tl.load(V_origin_ptr + q_high * stride_v4)

        gv = tl.load(GV_ptr + q_high * stride_v4)
        gv_normalizer = tl.load(GV + v_offset + (q_high + 15) * stride_v4 + tl.arange(0, BLOCK_DMODEL_V)).to(tl.float32)

        gv = tl.exp(gv_normalizer[None,:] - gv)
        dv = dv * gv
        tl.store(DV_ptr + q_high * stride_v4, dv.to(DV.dtype.element_ty))

        dgv_prev = tl.load(DGV_ptr + q_high * stride_v4)
        v = tl.load(V_origin_ptr + q_high * stride_v4)
        tl.store(DGV_ptr + q_high * stride_v4, (dgv_prev - dv * v).to(DGV.dtype.element_ty))
        
        
        
class FlashGRet_O(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, v, gv, chunk_size=16):
        assert gv.dtype == torch.float32
        # assert A.dtype == torch.float32

        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
       
        # assert gk.dtype == gv.dtype == torch.float32        
        BLOCK_M = BLOCK_N = v.shape[-2]

        # shape constraints
        Lv = v.shape[-1]
        BLOCK_V = min(128, Lv)
        ctx.BLOCK_V = BLOCK_V 

        assert v.shape[-1] % BLOCK_V == 0
        v_exp = v.clone()
        
        grid = (v.shape[2] , v.shape[0] * v.shape[1],  max(1, v.shape[-1] // BLOCK_V))
        o = torch.empty_like(v)            

        _fwd_compute_O[grid](A, v_exp, gv, o,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
            BLOCK_DMODEL_V=BLOCK_V, num_warps= 8 if BLOCK_V==128 else 4, num_stages=5
        )

        ctx.save_for_backward(A, v, v_exp, gv, o)
        ctx.grid = grid        
        ctx.chunk_size = chunk_size
        return o





    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        A, v, v_exp, gv, o = ctx.saved_tensors

        BLOCK_V = ctx.BLOCK_V
        assert v.shape[-1] % BLOCK_V == 0


        # dA = torch.empty_like(A)
        dv = torch.zeros_like(v)
        dgv = torch.zeros_like(gv)
        
        # for now.
        BLOCK_M = BLOCK_N = v.shape[-2]
        
        # shape constraints
        # Lv = v.shape[-1]
        # grid = (v.shape[2] , v.shape[0] * v.shape[1],  v.shape[-1] // BLOCK_V)
        grid = ctx.grid 

        dA = torch.empty(v.shape[-1] // BLOCK_V if BLOCK_V == 128 else 1, A.shape[0], A.shape[1], A.shape[2], A.shape[3], A.shape[3], device=A.device, dtype=A.dtype)

        _bwd_kernel_dav[grid](
            v_exp, v, gv, A, o, 
            do, dA,
            dv, dgv,
            v.shape[0], v.shape[1],
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,  
            BLOCK_DMODEL_V=ctx.BLOCK_V, num_warps=8, num_stages=4
        )        

        return dA.sum(0).to(A), dv.to(v), dgv.to(gv), None



def compute_inner_o2(qk, value, decay_value):
    # query = rearrange(query, 'b h (n c) d -> b h n c d', c=chunk_size)
    # key = rearrange(key, 'b h (n c) d -> b h n c d', c=chunk_size)
    # value = rearrange(value, 'b h (n c) d -> b h n c d', c=chunk_size)

    original_dtype = qk.dtype
    value = value.float()
    qk = qk.float()

    decay_value = decay_value.float().exp()
    
    value = value / decay_value 
    return ((qk @ value) * decay_value).to(original_dtype)


    

def compute_inner_fast(qk, value, decay_value):
    # query = rearrange(query, 'b h (n c) d -> b h n c d', c=chunk_size)
    # key = rearrange(key, 'b h (n c) d -> b h n c d', c=chunk_size)
    # value = rearrange(value, 'b h (n c) d -> b h n c d', c=chunk_size)
    # decay_value_exp = (decay_value[..., -1, None, :] - decay_value).exp()
    # value2 = value * decay_value_exp

    return FlashGRet_O.apply(A.contiguous(), value.contiguous(),  decay_value.contiguous())
    
    

if __name__ == "__main__":
    B = 32
    H = 4
    L = 2048
    D_QK = 256
    D_V = 512
    print("Hello.")

    requires_grad = True 
    chunk_size = 64
    num_chunk = L // chunk_size
    dtype = torch.bfloat16
    A2 = torch.randn(B, H, num_chunk, chunk_size, chunk_size, device='cuda').to(dtype).sigmoid()
    mask = torch.triu(torch.ones(chunk_size, chunk_size), diagonal=1).bool().to(A2.device)
    
    # for i in range(0, chunk_size, 32):
    #     mask[i:i+32, i:i+32] = True
        
    A2.requires_grad_(requires_grad)
    A = A2.masked_fill(mask, 0)

    v = (torch.rand(B, H, num_chunk, chunk_size, D_V, device='cuda')).to(dtype).requires_grad_(requires_grad)
    gk = torch.randn(B, H, num_chunk, chunk_size, D_V, device='cuda').requires_grad_(requires_grad)

    gk3 = F.logsigmoid(gk) / 16
    gk3 = gk3.cumsum(-2)


    # gv3 = F.logsigmoid(gv) / 32
    # gk3 = gk3.clamp(min=-5)
    # gv3 = gv3.clamp(min=-5)
    # gk = (gk3).cumsum(-2)
    # gv1 = (gv3).cumsum(-2) 
    # gk2 = (rearrange(gk3, 'b h (n c) d -> b h n c d', c=chunk_size)).cumsum(-2)
    # gv2 = (rearrange(gv3, 'b h (n c) d -> b h n c d', c=chunk_size)).cumsum(-2)
    # breakpoint()


    o = FlashGRet_O.apply(A, v, gk3)

    #gradient check

    o.sum().backward(retain_graph=True)

    target = [A2, v, gk]

    grad1= []
    grad2= []
    for s in target:
        grad1.append(s.grad.clone())
        s.grad.zero_()
    
    o2 = compute_inner_o2(A, v, gk3)    
    o2.sum().backward(retain_graph=True)

    # o3 = compute_inner_o2(A, v, gk3)

    for s in target:
        grad2.append(s.grad.clone())
        s.grad.zero_()
    
    print( (o - o2).abs().max())
    

    for a, b in zip(grad1, grad2):
        print( (a  - b).abs().max())


    breakpoint()

    # breakpoint()

    for _ in range(200):
        o = FlashGRet_O.apply(A, v, gk3)
        if requires_grad:
            o.sum().backward(retain_graph=True)    
        o2 = FlashGRet_O.apply(A, v, gk3)
        if requires_grad:
            o2.sum().backward(retain_graph=True)    

    print("warm up done.")
    print(o-o2)

    torch.cuda.synchronize()
    start = time.time()
    

    for _ in range(500):
        o = FlashGRet_O.apply(A, v, gk3)
        if requires_grad:
            o.sum().backward(retain_graph=True)    

    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}")

    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(500):
        o2 = FlashGRet_O.apply(A, v, gk)
        if requires_grad:
            o2.sum().backward(retain_graph=True)    

    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}")

    torch.cuda.synchronize()
    start = time.time()
    
    