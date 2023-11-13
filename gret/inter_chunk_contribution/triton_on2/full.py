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


# 输入 Q已经经过了GK的衰减。
# GK GV应该是全局的衰减的cumsum
# V 应该是已经经过了GV的衰减
@triton.jit
def _fwd_kernel(
    Q, K, V, GK, GV, 
    O, 
    stride_q1, stride_q2, stride_q3, stride_q4,
    stride_v1, stride_v2, stride_v3, stride_v4,
    Z, H, N_CTX, D,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr,
):
    # Q block的index
    start_m = tl.program_id(0)
    # Batch & Head 的index
    off_hz = tl.program_id(1)

    qk_offset = off_hz * stride_q2
    v_offset = off_hz * stride_v2

    Q_block_ptr = tl.make_block_ptr(
        base= Q + qk_offset,
        shape=(N_CTX, BLOCK_DMODEL_QK),
        strides=(stride_q3, stride_q4),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL_QK),
        order=(1, 0)
    )
    
    K_trans_block_ptr = tl.make_block_ptr(
        base= K + qk_offset,
        shape=(BLOCK_DMODEL_QK, N_CTX),
        strides=(stride_q4, stride_q3),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL_QK, BLOCK_N),
        order=(0, 1)
    )

    GK_trans_block_ptr = tl.make_block_ptr(
        base= GK + qk_offset,
        shape=(BLOCK_DMODEL_QK, N_CTX),
        strides=(stride_q4, stride_q3),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL_QK, BLOCK_N),
        order=(0, 1)
    )

    V_block_ptr = tl.make_block_ptr(
        base= V + v_offset,
        shape=(N_CTX, BLOCK_DMODEL_V),
        strides=(stride_v3, stride_v4),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL_V),
        order=(1, 0)
    )

        
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # initialize pointer to m and l
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=tl.float32)
    
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr) 

    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m) * BLOCK_N 

    prev_v_normalizer = tl.zeros([BLOCK_DMODEL_V,], dtype=tl.float32) + 1e9
    

    # shape: BLOCK_DMODLE_QK
    # 对应的是每个Q Block第一个的
    gk_normalizer = tl.load(GK + qk_offset + start_m * BLOCK_N * stride_q3 + tl.arange(0,BLOCK_DMODEL_QK)).to(tl.float32)

    V_normalizer_block_ptr = GV + v_offset + (BLOCK_N-1) * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)

    ## TODO: 感觉这里还是需要fix一下的
    ## TODO: 我需要考虑mask的问题吗？ 如果BLOCK_N和BLOCK_M不相等的话， 是有必要的。
    ## w/ lo and hi looks good 
    for _ in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k_trans = tl.load(K_trans_block_ptr).to(tl.float32)
        gk_trans = tl.load(GK_trans_block_ptr).to(tl.float32)
        
        # 考虑k上的衰减
        k_trans = k_trans * tl.exp(gk_normalizer[:, None] - gk_trans)
        # is BF16 enough? should we use fp32?
        qk = tl.dot(q, k_trans.to(q.dtype), allow_tf32=True) 
        
        # -- compute qk ---        
        v = tl.load(V_block_ptr)               

        v_normalizer = tl.load(V_normalizer_block_ptr).to(tl.float32)        
        output = tl.dot(qk.to(v.dtype), v, allow_tf32=True)        
        
        # 第一个case无所谓，所以这样也行吧
        acc = acc * tl.exp(v_normalizer - prev_v_normalizer)[None, :] + output
        
        prev_v_normalizer = v_normalizer        
        K_trans_block_ptr = tl.advance(K_trans_block_ptr, (0, BLOCK_N))
        GK_trans_block_ptr = tl.advance(GK_trans_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_M, 0))
        V_normalizer_block_ptr += BLOCK_N * stride_v3
        

    O_block_ptr = tl.make_block_ptr(
        base= O + v_offset,
        shape=(N_CTX, BLOCK_DMODEL_V),
        strides=(stride_v3, stride_v4),
        offsets=(start_m * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL_V),
        order=(1, 0)
    )

    # Q block 在 V 分量的衰减
    Q_GV_block_ptr = tl.make_block_ptr(
        base= GV + v_offset,
        shape=(N_CTX, BLOCK_DMODEL_V),
        strides=(stride_v3, stride_v4),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL_V),
        order=(1, 0)
    )

    v_normalizer = tl.load(Q_GV_block_ptr)
    acc = acc * tl.exp(v_normalizer - prev_v_normalizer[None, :]) 
    tl.store(O_block_ptr, acc.to(O.dtype.element_ty))    



class FlashGRet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, gk, gv, BLOCK_N = 32):
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        
        # for now.
        BLOCK_M = BLOCK_N
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        # right
        o = torch.empty_like(v)

        grid = (triton.cdiv(q.shape[2], BLOCK_N), q.shape[0] * q.shape[1], 1)

        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        

        _fwd_kernel[grid](
            q, k, v, gk, gv, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],            
            BLOCK_N=BLOCK_N, BLOCK_DMODEL_QK=Lk, BLOCK_M=BLOCK_M, 
            BLOCK_DMODEL_V=Lv,
        )
    

        ctx.save_for_backward(q, k, v, gk, gv)
        ctx.grid = grid
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_DMODEL_QK = Lk
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_DMODEL_V = Lv
        ctx.head = q.shape[1]
        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("FlashGRet backward not implemented yet")


def flash_gret_full(q, k, v, gk, gv, BLOCK_N = 64):   

    gk = gk.cumsum(-2)
    gv = gv.cumsum(-2)

    q = rearrange(q, 'b h (n c) d -> b h n c d', c=BLOCK_N)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=BLOCK_N)
    
    gk1 = rearrange(gk, 'b h (n c) d -> b h n c d', c=BLOCK_N) 
    gv2 = rearrange(gv, 'b h (n c) d -> b h n c d', c=BLOCK_N)
    
    gk_q_block_first_element = gk1[:, :, :, 0, :]
    gv_k_block_last_element = gv2[:, :, :, -1, :]

    # query 在 K 上的衰减，提前考虑进去
    gk1 = (gk1 - gk_q_block_first_element.unsqueeze(-2)).exp()
    q = q * gk1

    # value 在 V 上的衰减，提前考虑进去
    gv2 = (gv_k_block_last_element.unsqueeze(-2) - gv2).exp()
    v = v * gv2
    
    q = rearrange(q, 'b h n c d -> b h (n c) d')
    v = rearrange(v, 'b h n c d -> b h (n c) d')
                
    inter_chunk_contribution = FlashGRet.apply(q, k, v, gk, gv, BLOCK_N)

    output = inter_chunk_contribution
    
    return output 


