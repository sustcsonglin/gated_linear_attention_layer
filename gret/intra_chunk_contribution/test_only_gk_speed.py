
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
from cuda_chunk64x_dim64x.fn_only_gk import cuda_compute_intra
from triton_chunk16x.fn_only_gk import FlashGRet


def compute_inner(query, key, value, decay_key):
    mask = torch.triu(torch.ones(query.shape[-2], key.shape[-2]), diagonal=1).bool().to(query.device)
    original_dtype = query.dtype
    decay_key = decay_key.float().exp()
    
    query = query.float()
    key = key.float()

    query = (query * decay_key)
    key = key / decay_key 
    qk = (query @ key.transpose(-1, -2)).masked_fill_(mask, 0).to(original_dtype)
    return qk @ value 



if __name__ == "__main__":
    B = 8
    H = 4
    L = 2048
    D_QK = 256
    D_V = 512

    requires_grad = True
    chunk_size = 64
    num_chunk = L // chunk_size

    dtype = torch.bfloat16
    q = (torch.rand(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype)).requires_grad_(requires_grad)  
    k = torch.rand(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype).requires_grad_(requires_grad)
    v = torch.rand(B, H, num_chunk, chunk_size, D_V, device='cuda').to(dtype).requires_grad_(requires_grad)
    gk = torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda') .requires_grad_(requires_grad)
    # gv = torch.randn(B, H, L, D_V, device='cuda').requires_grad_(requires_grad)

    gk3 = F.logsigmoid(gk) / 16
    # gv3 = F.logsigmoid(gv) / 16

    gk3 = gk3.clamp(min=-5)
    # gv3 = gv3.clamp(min=-5)

    gk = (gk3).cumsum(-2) 
    # gv1 = (gv3).cumsum(-2) 

    # gk2 = (rearrange(gk3, 'b h (n c) d -> b h n c d', c=64)).cumsum(-2)
    # gv2 = (rearrange(gv3, 'b h (n c) d -> b h n c d', c=32)).cumsum(-2)

    # breakpoint()

    for _ in range(600):
        o = FlashGRet.apply(q, k, gk) @ v 
        o2 = compute_inner(q, k, v, gk)
                
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
    
    for _ in range(1000):
        o2 = compute_inner(q, k, v, gk)
        if requires_grad:
            o2.sum().backward(retain_graph=True)    
    
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
    
    for _ in range(1000):
        # q = rearrange(q, 'b h (n c) d -> b h n c d', c=64)
        # k = rearrange(k, 'b h (n c) d -> b h n c d', c=64)
        # v = rearrange(v, 'b h (n c) d -> b h n c d', c=64)
        # # gk = rearrange(gk, 'b h (n c) d -> b h n c d', c=64)
        o = cuda_compute_intra(q, k, gk) @ v
        if requires_grad:
            o.sum().backward(retain_graph=True)

    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}, fn:{compute_inner}")    
