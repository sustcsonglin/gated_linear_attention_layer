
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

def compute_inner(query, key,  decay_key):
    mask = torch.triu(torch.ones(query.shape[-2], key.shape[-2]), diagonal=1).bool().to(query.device)
    original_dtype = query.dtype
    decay_key = decay_key.float().exp()    

    query = query.float()
    key = key.float()

    query = (query * decay_key)
    key = key / decay_key 
    qk = (query @ key.transpose(-1, -2)).masked_fill_(mask, 0).to(original_dtype)
    return qk  




if __name__ == "__main__":
    B = 8
    H = 4
    L = 2048
    D_QK = 128
    D_V = 256


    requires_grad = True
    chunk_size = 64
    num_chunk = L // chunk_size

    dtype = torch.float32
    q = (torch.rand(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype)).requires_grad_(requires_grad)  
    k = torch.rand(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype).requires_grad_(requires_grad)
    v = torch.rand(B, H, num_chunk, chunk_size, D_V, device='cuda').to(dtype).requires_grad_(requires_grad)
    gk = torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda')  
    # gv = torch.randn(B, H, L, D_V, device='cuda').requires_grad_(requires_grad)

    gk3 = F.logsigmoid(gk) / 4
    # gv3 = F.logsigmoid(gv) / 16

    gk3 = gk3.clamp(min=-5)
    # gv3 = gv3.clamp(min=-5)

    gk = (gk3).cumsum(-2).requires_grad_(requires_grad)
    # gv1 = (gv3).cumsum(-2) 

    # gk2 = (rearrange(gk3, 'b h (n c) d -> b h n c d', c=64)).cumsum(-2)
    # gv2 = (rearrange(gv3, 'b h (n c) d -> b h n c d', c=32)).cumsum(-2)

    # breakpoint()

    # for _ in range(600):
    o = FlashGRet.apply(q, k, gk) 
    o2 = compute_inner(q, k,  gk)

    o = FlashGRet.apply(q, k, gk)
    o.sum().backward(retain_graph=True)

    target = [q, k, gk]


    grad1= []
    grad2= []
    for s in target:
        grad1.append(s.grad.clone())
        s.grad.zero_()
    
    o2 = compute_inner(q, k, gk)
    o2.sum().backward(retain_graph=True)

    for s in target:
        grad2.append(s.grad.clone())
        s.grad.zero_()
    
    print( (o - o2).abs().max())



    for a, b in zip(grad1, grad2):
        print( (a  - b).abs().max())

