import torch
from einops import rearrange
import triton 
import triton.language as tl
import torch.nn.functional as F
from src.intra_chunk_contribution.fn import intra_chunk_onc
from src.inter_chunk_contribution.fn import inter_chunk_onc
from time_counter import TimeCounter


def gated_linear_attention(q, k, v, gk, gv, normalizer_gk=16, normalizer_gv=16,  num_head=8, chunk_size=256):
    # assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert gk.dtype == gv.dtype == torch.float32
    q = rearrange(q, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
    k = rearrange(k, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
    v = rearrange(v, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
    gk = rearrange(gk, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
    gv = rearrange(gv, 'b (n c) (h d) -> b h n c d', h = num_head, c = chunk_size).contiguous()
    
    with TimeCounter.profile_time("p1"):
        gk, gv, o1 = inter_chunk_onc(q, k, v, gk, gv, normalizer_gk, normalizer_gv)
    with TimeCounter.profile_time("p2"):
        o2 = intra_chunk_onc(q, k, v, gk, gv)
    o = (o1 + o2)
    return rearrange(o, 'b h n c d -> b (n c) (h d)')



if __name__ == "__main__":
    B = 32
    H = 1
    L = 2048
    D_K = 1024 * H
    D_V = 2048 * H

    chunk_size = 256
    device = "cuda"
    requires_grad = True

    q = torch.randn(B, L, D_K, device=device).to(torch.bfloat16).requires_grad_(requires_grad)
    k = torch.randn(B, L, D_K, device=device).to(torch.bfloat16).requires_grad_(requires_grad)
    v = torch.randn(B, L, D_V, device=device).to(torch.bfloat16).requires_grad_(requires_grad)
    gk = torch.randn(B, L, D_K, device=device)
    gv = torch.randn(B, L, D_V, device=device)
    
    for _ in range(1000):
        o = gated_linear_attention(q, k, v, gk, gv, num_head=H, chunk_size=128)
        o.sum().backward(retain_graph=True)
    

