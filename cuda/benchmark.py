import torch
from torch.utils.cpp_extension import load
import time 


def compute_intra(query, key, value, decay_key, decay_value, mask):
    original_dtype = query.dtype


    query = query.float()
    key = key.float()
    value = value.float()

    decay_key = decay_key.float()
    decay_value = decay_value.float()

    decay_key = decay_key.exp()
    decay_value = decay_value.exp()
    query = (query * decay_key)
    key = key / decay_key


    qk = (query @ key.transpose(-1, -2)).masked_fill_(mask, 0)     
    value = value / decay_value
  
    return ((qk @ value) * decay_value).to(original_dtype), qk
    # return qk 



from fn_chunk128_dim128 import cuda_compute_intra_chunk128_d128


batch = 4
num_head = 4
num_chunk = 16
chunk_size = 64
d_head_k = 256
d_head_v = 256

requires_grad = False

mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device='cuda'), diagonal=1)


Q = torch.randn(batch, num_head, num_chunk, chunk_size, d_head_k, device='cuda', dtype=torch.float32).requires_grad_(requires_grad)

K = torch.randn(batch, num_head, num_chunk, chunk_size, d_head_k, device='cuda', dtype=torch.float32).requires_grad_(requires_grad)

V = torch.randn(batch, num_head, num_chunk, chunk_size, d_head_v, device='cuda', dtype=torch.float32).requires_grad_(requires_grad)

g_A = torch.randn(batch, num_head, num_chunk, chunk_size, d_head_k, device='cuda').uniform_(0.8, 0.99).log()
g_A_cumsum = g_A.cumsum(-2)
g_A = g_A_cumsum.detach().clone().contiguous().requires_grad_(requires_grad)

g_B = torch.randn(batch, num_head, num_chunk, chunk_size, d_head_v, device='cuda').uniform_(0.8, 0.99).log()
g_B_cumsum = g_B.cumsum(-2)
g_B = g_B_cumsum.detach().clone().contiguous().requires_grad_(requires_grad)



for _ in range(100):
    
    C, _ = cuda_compute_intra_chunk128_d128(Q, K, V, g_A, g_B)
    if requires_grad:
        C.sum().backward(retain_graph=True)
    
    C2, _ = compute_intra(Q, K, V, g_A, g_B, mask)
    if requires_grad:
        C2.sum().backward(retain_graph=True)

    # _ = compute_intra(Q, K, V, g_A, g_B, mask)


# grad_C = torch.rand_like(C)
# C.backward(grad_C, retain_graph=True)
# concerned_paras = [g_A, g_B, Q, K, V]
# grad1 = []
# for p in concerned_paras:
#     grad1.append(p.grad.clone())
#     p.grad.zero_()

# D.backward(grad_C, retain_graph=True)

# for idx, p in enumerate(concerned_paras):
#     print( (grad1[idx] - p.grad).abs().max())

# breakpoint()





torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    C, _  = cuda_compute_intra_chunk128_d128(Q, K, V, g_A, g_B)
    if requires_grad:
        C.sum().backward(retain_graph=True)


torch.cuda.synchronize()
print("CUDA")
print("Time:", time.time() - start)
print("Max GPU memopry:", torch.cuda.max_memory_allocated() // 1024 // 1024)

torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
start = time.time()

for _ in range(1000):
    C, _  = compute_intra(Q, K, V, g_A, g_B, mask)
    if requires_grad:
        C.sum().backward(retain_graph=True)

    # C.sum().backward(retain_graph=True)

torch.cuda.synchronize()
print("Pytorch")
print("Time:", time.time() - start)
print("Max GPU memopry:", torch.cuda.max_memory_allocated() // 1024 // 1024)

