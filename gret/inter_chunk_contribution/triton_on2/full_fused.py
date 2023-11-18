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

### what the input size should be.
### B * H * L * D instead of B * H * NUM_CHUNK * CHUNK_SIZE * D.
@triton.jit
def _fwd_kernel(
    Q, K, V, GK, GV, O,
    stride_q1, stride_q2, stride_q3, stride_q4,
    stride_v1, stride_v2, stride_v3, stride_v4,
    # Z, H, N_CTX, D,
    BLOCK_DMODEL_QK: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    start_m = tl.program_id(0)

    if start_m == 0:
        return 

    off_hz = tl.program_id(1)

    qk_offset = off_hz * stride_q2
    v_offset = off_hz * stride_v2
    
    # initialize pointer to m and l
    # load q: it will stay in SRAM throughout
    # loop over k, v and update accumulator
    # lo = max(0, (start_m - 4) * BLOCK_Q)
    lo = 0

    hi = (start_m) * BLOCK_Q
    Q_ptr = Q + qk_offset + (start_m) * BLOCK_Q * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, BLOCK_Q)[:, None] * stride_q3
    q = tl.load(Q_ptr)
    q_gk = tl.load(GK + qk_offset + (start_m) * BLOCK_Q * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, BLOCK_Q)[:, None] * stride_q3)
    gk_normalizer = tl.load(GK + qk_offset + (start_m) * BLOCK_Q * stride_q3 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)

    q = q * tl.exp(q_gk - gk_normalizer[None, :]).to(q.dtype)

    K_trans_ptr = K + qk_offset + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, BLOCK_K)[None, :] * stride_q3
    GK_trans_ptr = GK + qk_offset + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, BLOCK_K)[None, :] * stride_q3
    
    V_ptr = V + v_offset + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, BLOCK_K)[ :, None] * stride_v3
    GV_ptr = GV + v_offset + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, BLOCK_K)[ :, None] * stride_v3

    q_gv_normalizer = tl.load(GV + v_offset + (start_m) * BLOCK_Q * stride_v3 + tl.arange(0,BLOCK_DMODEL_V)).to(tl.float32)
                        
    acc = tl.zeros([BLOCK_Q, BLOCK_DMODEL_V], dtype=tl.float32)

    for _ in range(lo, hi, BLOCK_K):
        # -- load k, v --
        k_trans = tl.load(K_trans_ptr)
        normalizer2 = tl.load(GK + qk_offset + (start_m) * BLOCK_Q * stride_q3 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)        

        k_trans = k_trans * tl.exp(normalizer2 - gk_normalizer[:, None]).to(k_trans.dtype)[:, None]                
        qk = tl.dot(q, k_trans, allow_tf32=False) 
                
        v = tl.load(V_ptr) 
        acc += tl.dot(qk.to(v.dtype), v, allow_tf32=False)
        
        K_trans_ptr += BLOCK_K * stride_q3
        GK_trans_ptr += BLOCK_K * stride_q3
        V_ptr += BLOCK_K * stride_v3
        GV_ptr += BLOCK_K * stride_v3
    
    q_gv = tl.load(GV + v_offset + ((start_m) * BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None])* stride_v3 + tl.arange(0,BLOCK_DMODEL_V)[None, :])
    q_gv = tl.exp(q_gv - q_gv_normalizer[None, :])
    acc *= q_gv.to(acc.dtype)
    tl.store(O + v_offset + ((start_m) * BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None])* stride_v3 + tl.arange(0,BLOCK_DMODEL_V)[None, :], acc.to(O.dtype.element_ty))
    


    
        
class InterChunk_On2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, gk, gv, BLOCK_Q = 128, BLOCK_K=32):
        assert q.is_contiguous()
        assert k.is_contiguous()
        assert gk.is_contiguous()
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")

        o = torch.empty_like(v)

        # for now.
        # BLOCK_Q = BLOCK_K
        # shape constraints
        # Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        # right
        # o = torch.empty_like(v)

        # A = torch.zeros(q.shape[0], q.shape[1], q.shape[2], q.shape[2], device=q.device, dtype=q.dtype)

        grid = (triton.cdiv(q.shape[2], BLOCK_Q), q.shape[0] * q.shape[1], 1)

        # assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        
        _fwd_kernel[grid](
            q, k, v, gk, gv, o, 
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            # q.shape[0], q.shape[1], q.shape[2], q.shape[3],            
            BLOCK_K=BLOCK_K, BLOCK_DMODEL_QK=q.shape[-1], BLOCK_Q=BLOCK_Q,  BLOCK_DMODEL_V=v.shape[-1],
            num_warps=8, num_stages=8
            # BLOCK_DMODEL_V=Lv,
        )



        # ctx.save_for_backward(q, k, gk, A)
        ctx.grid = grid
        return o




    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("FlashGRet backward not implemented yet")


def compute_inter_chunk_on2_fused(q, k, v, gk, gv, chunk_size):

    gk = gk.cumsum(-2)
    gv = gv.cumsum(-2)

    O = InterChunk_On2.apply(q, k, v, gk, gv, 16, 16)

    return O 


