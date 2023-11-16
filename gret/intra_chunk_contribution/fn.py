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
from triton_chunk16x.fn_only_gk_no_intra import FlashGRet
from triton_chunk16x.fn_only_gv_no_intra import FlashGRet_O

def intra_chunk_computation_pure_triton(q, k, v, gk, gv, chunk_size=None):
    # if chunk_size is not None, we need manually chunk the input into multiple chunks.
    # if is None, meaning that the input is already chunked.
    if chunk_size is not None:
        q = rearrange(q, 'b h (n c) d  -> b h n c d', c=chunk_size)
        k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
        v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
        gk = rearrange(gk, 'b h (n c) d -> b h n c d', c=chunk_size)
        gv = rearrange(gv, 'b h (n c) d -> b h n c d', c=chunk_size)

    chunk_size = v.shape[-2]
    ### assert chunk_size is multiple of 16.
    assert chunk_size % 16 == 0
    # the large chunk size is multiple times of 16. the small chunk size is always 16 because we want to use the fast and numerically stable cuda kernel for 16x16 computation 
    # TODO: implement a 32*32 one?

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    multiple = chunk_size // 16    
    A = FlashGRet.apply(q, k, gk)
    A = A.masked_fill_(mask, 0)
    inner_chunk_contribution = FlashGRet_O.apply(A, v, gv)
    return inner_chunk_contribution


def compute_inner(query, key, value, decay_key, decay_value):
    # query = rearrange(query, 'b h (n c) d -> b h n c d', c=chunk_size)
    # key = rearrange(key, 'b h (n c) d -> b h n c d', c=chunk_size)
    # value = rearrange(value, 'b h (n c) d -> b h n c d', c=chunk_size)

    mask = torch.triu(torch.ones(query.shape[-2], key.shape[-2]), diagonal=1).bool().to(query.device)

    original_dtype = query.dtype
    decay_key = decay_key.double().exp()
    decay_value = decay_value.double().exp()    

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

    require_grad= False

    print("Hello")


    dtype = torch.bfloat16
    q = torch.rand(B, H, L, D_QK, device='cuda').to(dtype).requires_grad_(require_grad)
    k = torch.rand(B, H, L, D_QK, device='cuda').to(dtype).requires_grad_(require_grad)
    v = torch.rand(B, H, L, D_V, device='cuda').to(dtype).requires_grad_(require_grad)
    gk = torch.randn(B, H, L, D_QK, device='cuda').to(dtype) 
    gv = torch.randn(B, H, L, D_V, device='cuda').to(dtype) 

    gk = (F.logsigmoid(gk) / 16).cumsum(-2).requires_grad_(require_grad)
    gv = (F.logsigmoid(gv) / 16).cumsum(-2).requires_grad_(require_grad)



    for _ in range(100):
        # o = naive(q, k, v, gk, gv, BLOCK_N = 64)
        o2 = intra_chunk_computation_pure_triton(q, k, v, gk, gv, chunk_size = 64)
        o2 = intra_chunk_computation_pure_triton(q, k, v, gk, gv, chunk_size = 16)
        o2 = intra_chunk_computation_pure_triton(q, k, v, gk, gv, chunk_size = 32)
        o2 = intra_chunk_computation_pure_triton(q, k, v, gk, gv, chunk_size = 128)
        o2 = intra_chunk_computation_pure_triton(q, k, v, gk, gv, chunk_size = 256)
        o2 = intra_chunk_computation_pure_triton(q, k, v, gk, gv, chunk_size = 512)
        o2 = intra_chunk_computation_pure_triton(q, k, v, gk, gv, chunk_size = 1024)
        # o3 = flash_gret_full(q, k, v, gk, gv, BLOCK_N = 64)
    
    print("warm up done.")

    for chunk_size in [16, 32, 64, 128, 256, 512, 1024]:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(200):
            o2 = intra_chunk_computation_pure_triton(q, k, v, gk, gv, chunk_size = chunk_size)
            if require_grad:
                o2.sum().backward(retain_graph=True)    
        torch.cuda.synchronize()
        end = time.time()
        print(f"scan gret onc, time:{end - start},  chunk size: {chunk_size}")      
    
    
