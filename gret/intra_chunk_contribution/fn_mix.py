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
from triton_chunk16x.fn_only_gk import FlashGRet
from triton_chunk16x.fn_only_gv import FlashGRet_O
from cuda_chunk16_dim64x.fn import cuda_compute_intra

def intra_chunk_computation_mixed_cuda_triton(q, k, v, gk, gv):
    chunk_size = v.shape[-2]
    ### assert chunk_size is multiple of 16.
    assert chunk_size % 16 == 0
    # the large chunk size is multiple times of 16. the small chunk size is always 16 because we want to use the fast and numerically stable cuda kernel for 16x16 computation 
    # TODO: implement a 32*32 one?

    multiple = chunk_size // 16    
    A = FlashGRet.apply(q, k, gk)
    inner_chunk_contribution = FlashGRet_O.apply(A, v, gv)

    q = rearrange(q, 'b h n (s c) d -> b h (n s) c d', c = 16)
    k = rearrange(k, 'b h n (s c) d -> b h (n s) c d', c = 16)
    v = rearrange(v, 'b h n (s c) d -> b h (n s) c d', c = 16)
    gk = rearrange(gk, 'b h n (s c) d -> b h (n s) c d', c = 16)
    gv = rearrange(gv, 'b h n (s c) d -> b h (n s) c d', c = 16)
    inner_chunk_contribution2 = cuda_compute_intra(q, k, v, gk, gv)
    inner_chunk_contribution += rearrange(inner_chunk_contribution2, 'b h (n s) c d -> b h n (s c) d', s = multiple)
    
    return inner_chunk_contribution



if __name__ == "__main__":
    B = 2
    H = 4
    L = 512
    D_QK = 128
    D_V = 128

    requires_grad = False 
    chunk_size = 64
    num_chunk = L // chunk_size

    dtype = torch.bfloat16
    q = (torch.rand(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype)).requires_grad_(requires_grad)  
    k = torch.rand(B, H, num_chunk, chunk_size, D_QK, device='cuda').to(dtype).requires_grad_(requires_grad)
    v = torch.rand(B, H, num_chunk, chunk_size, D_V, device='cuda').to(dtype).requires_grad_(requires_grad)
    gk = torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda').requires_grad_(requires_grad)  
    gv =  torch.randn(B, H, num_chunk, chunk_size, D_QK, device='cuda').requires_grad_(requires_grad)  

    gk = F.logsigmoid(gk) / 8
    gv = F.logsigmoid(gv) / 8

    gk = gk.cumsum(-2)
    gv = gv.cumsum(-2)

    # gk3 = gk3.clamp(min=-5)
    # gv3 = gv3.clamp(min=-5)


    output = intra_chunk_computation_mixed_cuda_triton(q, k, v, gk, gv)
    breakpoint()




