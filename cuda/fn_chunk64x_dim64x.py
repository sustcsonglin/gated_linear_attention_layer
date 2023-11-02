import torch 
import os 
from torch.utils.cpp_extension import load
module_path = os.path.dirname(__file__)


cuda_compute_inner = load(
    name="cuda_compute_inner_chunk64x_d64x",
    sources=[os.path.join(module_path, "kernel_chunk64x_dim64x.cpp"), os.path.join(module_path, "kernel_chunk64x_dim64x.cu")],
    # extra_cuda_cflags=["-arch=sm_70"],  # Set the right compute capability based on your GPU
    verbose=True,
)


import torch.nn.functional as F


class CUDA_inner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, gk):
        original_dtype = q.dtype
        assert q.shape[-1] % 64 == 0

        q = q.float().contiguous()
        k = k.float().contiguous()
        gk = gk.float().contiguous()

        qk = cuda_compute_inner.forward(q, k, gk)
    
        ctx.save_for_backward(q, k, gk)
        ctx.orig_dtype = original_dtype        
        
    
        return qk.to(original_dtype)

    @staticmethod
    def backward(ctx, dqk):
        
        orig_dtype =  ctx.orig_dtype        

        dqk = dqk.float().contiguous()

        q, k, gk = ctx.saved_tensors
                
        dq, dk, dgk = cuda_compute_inner.backward(q, k, gk, dqk)

        return dq.to(orig_dtype), dk.to(orig_dtype), dgk.to(orig_dtype)


cuda_compute_intra_chunk64x_d64x = CUDA_inner.apply

def naive(q, k, gk):
    b, h, num_chunk, chunk_size, d = q.shape
    chunk_size = q.shape[-2]

    o = torch.zeros_like(q)
    A = torch.zeros(b, h, num_chunk, chunk_size, chunk_size, device='cuda')

    for i in range(chunk_size):
        q_i = q[:, :, :, i]
        g_i = gk[:, :, :, i]
        o_i = torch.zeros_like(q_i)

        for j in range(0, i+1):
            k_j = k[:, :, :, j]
            
            g_j = gk[:, :, :, j]

            o_ij = (q_i * k_j * (g_i - g_j).exp()).sum(-1)

            A[:, :, :, i, j] = o_ij

            # o_i += o_ij[..., None] * v[:, :, :, j] 
    
        # o[:, :, :, i] = o_i                 
    return A 

                



if __name__ == "__main__":
    B = 1
    H = 1
    L = 256
    D = 128
    CHUNK_SIZE = 256
    NUM_CHUNK = L // CHUNK_SIZE
    
    requires_grad = True
    q = torch.rand((B, H, NUM_CHUNK, CHUNK_SIZE, D), device='cuda:0').requires_grad_(requires_grad)
    k = torch.rand((B, H, NUM_CHUNK, CHUNK_SIZE, D), device='cuda:0').requires_grad_(requires_grad)
    gk = F.logsigmoid(torch.rand((B, H, NUM_CHUNK, CHUNK_SIZE, D), device='cuda:0'))   
    gk = gk.cumsum(-2).requires_grad_(requires_grad)

    mask = torch.ones(CHUNK_SIZE, CHUNK_SIZE, device='cuda:0').triu(1).bool()
    A = cuda_compute_intra_chunk64x_d64x(q, k, gk)

    # output.masked_fill_(mask, 0)

    A2 = naive(q, k,  gk)




    # assert torch.isclose(output, output2, atol=1e-3).all()
    assert torch.isclose(A, A2, atol=1e-3).all()

    A.sum().backward()
    k_grad_clone = k.grad.clone()
    q_grad_clone = q.grad.clone()
    gk_grad_clone = gk.grad.clone()
    
    v.grad.zero_()
    gv.grad.zero_()
    k.grad.zero_()
    q.grad.zero_()
    gk.grad.zero_()

    output2.sum().backward()
    
    assert torch.isclose(v.grad, v_grad_clone, atol=1e-3).all()   
    assert torch.isclose(gv.grad, gv_grad_clone, atol=1e-3).all()
    assert torch.isclose(k.grad, k_grad_clone, atol=1e-3).all()
    assert torch.isclose(q.grad, q_grad_clone, atol=1e-3).all()
    assert torch.isclose(gk.grad, gk_grad_clone, atol=1e-3).all()









    

    



        
        

