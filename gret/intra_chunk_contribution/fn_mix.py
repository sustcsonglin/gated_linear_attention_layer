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
from cuda_chunk16_dim64x.fn import cuda_compute_intra

def intra_chunk_computation_mixed_cuda_triton(q, k, v, gk, gv):
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

    q = rearrange(q, 'b h n (s c) d -> b h (n s) c d', c = 16) 
    k = rearrange(k, 'b h n (s c) d -> b h (n s) c d', c = 16) 
    v = rearrange(v, 'b h n (s c) d -> b h (n s) c d', c = 16) 
    gk = rearrange(gk, 'b h n (s c) d -> b h (n s) c d', c = 16) 
    gv = rearrange(gv, 'b h n (s c) d -> b h (n s) c d', c = 16) 
    inner_chunk_contribution2 = cuda_compute_intra(q, k, v, gk, gv) 
    inner_chunk_contribution = inner_chunk_contribution + rearrange(inner_chunk_contribution2, 'b h (n s) c d -> b h n (s c) d', s = multiple) 

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
    L = 64
    D_QK = 128
    D_V = 128

    requires_grad = True
    chunk_size = 64
    num_chunk = L // chunk_size


    dtype = torch.bfloat16
    q = (torch.rand(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype)).requires_grad_(requires_grad)  
    k = torch.rand(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype).requires_grad_(requires_grad)
    v = torch.rand(B, H, num_chunk, chunk_size, D_V, device='cuda').to(dtype).requires_grad_(requires_grad)
    gk = torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype) 
    gv =  torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype)

    gk = F.logsigmoid(gk) / 16
    gv = F.logsigmoid(gv) / 16


    gk = gk.cumsum(-2).requires_grad_(requires_grad)
    gv = gv.cumsum(-2).requires_grad_(requires_grad)


    # gk3 = gk3.clamp(min=-5)
    # gv3 = gv3.clamp(min=-5)     
     
    output = intra_chunk_computation_mixed_cuda_triton(q, k, v, gk, gv)

    output.sum().backward(retain_graph=True)

    target = [q, k, v, gk, gv]

    grad1= []
    grad2= []
    for s in target:
        grad1.append(s.grad.clone())
        s.grad.zero_()
    
    o2 = compute_inner(q, k, v, gk, gv)
    o2.sum().backward(retain_graph=True)

    for s in target:
        grad2.append(s.grad.clone())
        s.grad.zero_()
    
    print( (output - o2).abs().max())

    for a, b in zip(grad1, grad2):
        print( (a  - b).abs().max())


