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
def _fwd_kernel(
    A, V, GV, 
    O, 
    stride_a1, stride_a2, stride_a3, stride_a4,
    stride_v1, stride_v2, stride_v3, stride_v4,
    # Z, H, N_CTX, D,
    # BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_V: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr,
):
    start_m = tl.program_id(0)
    
    # # no intra chunk contribution
    if start_m == 0:
        return 

    off_hz = tl.program_id(1)
    a_offset = off_hz * stride_a2
    v_offset = off_hz * stride_v2

    # initialize pointer to m and l
    acc = tl.zeros([BLOCK_O, BLOCK_DMODEL_V], dtype=tl.float32)
    
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m) * BLOCK_O
    
    # starting from the first block.
    V_ptr = V + v_offset + tl.arange(0, BLOCK_V)[:, None] * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :]
    GV_ptr = GV + v_offset + tl.arange(0, BLOCK_V)[:, None] * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :]
    # the first element...
    q_gv_normalizer = tl.load(GV + v_offset + tl.arange(0, BLOCK_DMODEL_V) + start_m * BLOCK_O * stride_v3).to(tl.float32)
    A_ptr = A + a_offset + (start_m * BLOCK_O + tl.arange(0, BLOCK_O)[:, None]) * stride_a3 + tl.arange(0, BLOCK_V)[None, :]
    
    for _ in range(lo, hi, BLOCK_V):        
        qk = tl.load(A_ptr)                 
        # -- compute qk ---        
        v = tl.load(V_ptr)               
        gv = tl.load(GV_ptr)
        v = v * tl.exp(q_gv_normalizer[None, :] - gv).to(v.dtype) 
        output = tl.dot(qk, v, allow_tf32=False)                
        acc += output
        
        A_ptr += BLOCK_V 
        V_ptr += BLOCK_V * stride_v3
        GV_ptr += BLOCK_V * stride_v3


    GV_ptr = GV + v_offset + (start_m * BLOCK_O + tl.arange(0, BLOCK_O)[:, None]) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :]
    acc = acc * tl.exp(tl.load(GV_ptr) - q_gv_normalizer[None, :])

    O_ptr = O + v_offset + (start_m * BLOCK_O + tl.arange(0, BLOCK_O)[:, None]) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :]
    tl.store(O_ptr, acc.to(O.dtype.element_ty))        

class InterChunk_Compute_O(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, v, gv, BLOCK_O = 128, BLOCK_V = 32):
        assert A.is_contiguous()
        assert v.is_contiguous()
        assert gv.is_contiguous()        

        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")

        # for now.
        o = torch.zeros_like(v) 
        # num_chunk: 
        grid = (triton.cdiv(v.shape[2], BLOCK_O), v.shape[0] * v.shape[1], 1)

        # assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        
        _fwd_kernel[grid](
            A, v, gv, o,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            BLOCK_O=BLOCK_O, BLOCK_V=BLOCK_V,
            BLOCK_DMODEL_V=v.shape[-1],  num_warps=16, num_stages=8
        )
        ctx.save_for_backward(A, v, gv, o)
        ctx.grid = grid
        return o




    



    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("FlashGRet backward not implemented yet")


# def flash_gret_full(q, k, v, gk, gv, BLOCK_N = 32):   

#     gk = gk.cumsum(-2)
#     gv = gv.cumsum(-2)

#     q = rearrange(q, 'b h (n c) d -> b h n c d', c=BLOCK_N)
#     v = rearrange(v, 'b h (n c) d -> b h n c d', c=BLOCK_N)
    
#     gk1 = rearrange(gk, 'b h (n c) d -> b h n c d', c=BLOCK_N) 
#     gv2 = rearrange(gv, 'b h (n c) d -> b h n c d', c=BLOCK_N)
    
#     gk_q_block_first_element = gk1[:, :, :, 0, :]
#     gv_k_block_last_element = gv2[:, :, :, -1, :]

#     # query 在 K 上的衰减，提前考虑进去
#     gk1 = (gk1 - gk_q_block_first_element.unsqueeze(-2)).exp()
#     q = q * gk1

#     # value 在 V 上的衰减，提前考虑进去
#     gv2 = (gv_k_block_last_element.unsqueeze(-2) - gv2).exp()
#     v = v * gv2
    
#     q = rearrange(q, 'b h n c d -> b h (n c) d')
#     v = rearrange(v, 'b h n c d -> b h (n c) d')
                
#     inter_chunk_contribution = FlashGRet.apply(q, k, v, gk, gv, BLOCK_N)

#     output = inter_chunk_contribution
    
#     return output 


