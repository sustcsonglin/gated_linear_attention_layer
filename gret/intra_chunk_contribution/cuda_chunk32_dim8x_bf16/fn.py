import torch 
import os 
from torch.utils.cpp_extension import load
module_path = os.path.dirname(__file__)

import time 


cuda_compute_inner = load(
    name="cuda_compute_inner_bf16",
    sources=[os.path.join(module_path, "kernel_chunk32_dim8x.cpp"), os.path.join(module_path, "kernel_chunk32_dim8x.cu")],
    # extra_cuda_cflags=["-arch=sm_70"],  # Set the right compute capability based on your GPU
    verbose=True,
)

class CUDA_inner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, gk, gv):
        original_dtype = q.dtype

        assert q.shape[-1] % 8 == 0
        assert v.shape[-1] % 8 == 0
        assert q.shape[-2] == 32

        # we want to make sure that the input is in bf16
        # but the gate is in fp32 to maintain precision
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        assert gk.dtype == gv.dtype == torch.float32

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        gk = gk.contiguous()
        gv = gv.contiguous()
    
        output, qk = cuda_compute_inner.forward(q, k, v, gk, gv)
        ctx.save_for_backward(q, k, v, gk, gv, qk)
        ctx.orig_dtype = original_dtype                
        return output.to(original_dtype)


    @staticmethod
    def backward(ctx, do):
        orig_dtype =  ctx.orig_dtype
        do = do.contiguous()

        q, k, v, gk, gv, qk = ctx.saved_tensors
        dq, dk, dv, dgk, dgv = cuda_compute_inner.backward(q, k, v, gk, gv, qk, do)        
        return dq, dk, dv,  dgk, dgv



cuda_compute_intra = CUDA_inner.apply


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



if __name__ == "__main__":
    batch = 32
    num_head = 8
    num_chunk = 128
    chunk_size = 32
    d_head = 256
    dtype = torch.bfloat16
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device='cuda'), diagonal=1)
    Q = torch.randn(batch, num_head, num_chunk, chunk_size, d_head, device='cuda', dtype=dtype).requires_grad_(True)
    K = torch.randn(batch, num_head, num_chunk, chunk_size, d_head, device='cuda', dtype=dtype).requires_grad_(True)
    V = torch.randn(batch, num_head, num_chunk, chunk_size, d_head, device='cuda', dtype=dtype).requires_grad_(True)
    g_A = torch.randn(batch, num_head, num_chunk, chunk_size, d_head, device='cuda').uniform_(0.98, 0.99).log()
    g_A_cumsum = g_A.cumsum(-2)
    g_A = g_A_cumsum.detach().clone().contiguous().requires_grad_(True)

    g_B = torch.randn(batch, num_head, num_chunk, chunk_size, d_head, device='cuda').uniform_(0.98, 0.99).log()
    g_B_cumsum = g_B.cumsum(-2)
    g_B = g_B_cumsum.detach().clone().contiguous().requires_grad_(True)

    # for _ in range(10):
    C = cuda_compute_intra(Q, K, V, g_A, g_B)
    D, qk2 = compute_intra(Q, K, V, g_A, g_B, mask)
    


    grad_C = torch.rand_like(C)
    C.backward(grad_C, retain_graph=True)

    concerned_paras = [g_A, g_B, Q, K, V]

    grad1 = []

    for p in concerned_paras:
        grad1.append(p.grad.clone())
        p.grad.zero_()

    D.backward(grad_C, retain_graph=True)

    grad2 = []

    for idx, p in enumerate(concerned_paras):
        grad2.append(p.grad)
        print( (grad1[idx] - p.grad).abs().max())
    

    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(200):
        C = cuda_compute_intra(Q, K, V, g_A, g_B)
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
            
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(200):
        C = cuda_compute_intra(Q, K, V, g_A, g_B)
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



