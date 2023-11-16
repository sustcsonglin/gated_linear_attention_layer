import torch
from torch.utils.cpp_extension import load
import time 

@torch.jit.script
def compute_intra(query, key, value, decay_key, decay_value, mask):
    original_dtype = query.dtype
    
    decay_key = decay_key.float().exp()
    decay_value = decay_value.float().exp()

    query = query.float()
    key = key.float()
    value = value.float()
    
    query = (query * decay_key)
    key = key / decay_key

    qk = (query @ key.transpose(-1, -2)).masked_fill_(mask, 0)     
    value = value / decay_value
  
    return ((qk @ value) * decay_value).to(original_dtype), qk

from fn import cuda_compute_intra


batch = 4
num_head = 4
num_chunk = 128
chunk_size = 16
d_head = 128

mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device='cuda'), diagonal=1)

Q = torch.randn(batch, num_head, num_chunk, chunk_size, d_head, device='cuda', dtype=torch.float32).requires_grad_(True)

K = torch.randn(batch, num_head, num_chunk, chunk_size, d_head, device='cuda', dtype=torch.float32).requires_grad_(True)

V = torch.randn(batch, num_head, num_chunk, chunk_size, d_head, device='cuda', dtype=torch.float32).requires_grad_(True)


g_A = torch.randn(batch, num_head, num_chunk, chunk_size, d_head, device='cuda').uniform_(0.97, 0.99).log()
g_A_cumsum = g_A.cumsum(-2)
g_A = g_A_cumsum.detach().clone().contiguous().requires_grad_(True)

g_B = torch.randn(batch, num_head, num_chunk, chunk_size, d_head, device='cuda').uniform_(0.97, 0.99).log()
g_B_cumsum = g_B.cumsum(-2)
g_B = g_B_cumsum.detach().clone().contiguous().requires_grad_(True)


# for _ in range(10):
C, qk1 = cuda_compute_intra(Q, K, V, g_A, g_B)
D, qk2 = compute_intra(Q, K, V, g_A, g_B, mask)



grad_C = torch.rand_like(C)
C.backward(grad_C, retain_graph=True)

concerned_paras = [g_A, g_B, Q, K, V]

grad1 = []

for p in concerned_paras:
    grad1.append(p.grad.clone())
    p.grad.zero_()

D.backward(grad_C, retain_graph=True)

for idx, p in enumerate(concerned_paras):
    print( (grad1[idx] - p.grad).abs().max())

breakpoint()






torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
start = time.time()
for _ in range(200):
    C, qk1 = cuda_compute_intra(Q, K, V, g_A, g_B)
    C.sum().backward(retain_graph=True)
torch.cuda.synchronize()
print("CUDA")
print("Time:", time.time() - start)
print("Max GPU memopry:", torch.cuda.max_memory_allocated() // 1024 // 1024)


torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
start = time.time()
for _ in range(200):
    C, qk1 = compute_intra(Q, K, V, g_A, g_B, mask)

    C.sum().backward(retain_graph=True)
torch.cuda.synchronize()
print("Pytorch")
print("Time:", time.time() - start)
print("Max GPU memopry:", torch.cuda.max_memory_allocated() // 1024 // 1024)
