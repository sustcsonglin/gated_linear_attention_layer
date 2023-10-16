import torch
# from .triton_on2 import flash_rnn_on2
from einops import rearrange
import triton 
import triton.language as tl
from cuda import cuda_compute_intra
from pytorch_chunk_onc import torch_chunk_parallel_onc 
from fused_onc import fused_chunk_parallel_onc
from triton_flashattn import attention 


if __name__ == "__main__":
    B = 4
    H = 8
    L = 2048
    D = 128
    chunk_size = 16
    device = "cuda"
    requires_grad = False 
    dtype = torch.float16

    v1 = (torch.randn(B, H,  L, D) ).cuda().to(dtype).requires_grad_(requires_grad)
    v2 = (torch.randn(B, H, L, D) ).cuda().to(dtype).requires_grad_(requires_grad) 
    g1 = torch.randn(B, H,  L, D).cuda().uniform_(0.9, 0.99).log().to(dtype).requires_grad_(requires_grad)
    g2 = torch.randn(B, H, L, D).cuda().uniform_(0.9, 0.99).log().to(dtype).requires_grad_(requires_grad)
    # g1 = torch.zeros(B, H, L, D)
    # g2 = torch.zeros(B, H, L, D)
    q = (torch.randn(B, H, L, D) * 5).cuda().to(dtype).requires_grad_(requires_grad)  


    for _ in range(100):
        output1 = attention(q, v1, v2, True, 1)

        output2 =  torch_chunk_parallel_onc(v1, v2, g1, g2, q, chunk_size=chunk_size,use_triton=True)

    print("warm up done")



    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        output1 = (q @ v1.transpose(-1,-2)).softmax(-1) @ v2
    torch.cuda.synchronize()
    end = time.time()
    print("naive attention time", end - start)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        output1 = attention(q, v1, v2, True, 1)
    torch.cuda.synchronize()
    end = time.time()
    print("triton attention time", end - start)

    torch.cuda.synchronize()

    start = time.time()
    for _ in range(1000):
        output2 = torch_chunk_parallel_onc(v1, v2, g1, g2, q, chunk_size=chunk_size,use_triton=True, use_cuda=True)
    torch.cuda.synchronize()
    end = time.time()
    print("gated retnet time w/ cuda", end - start)

    torch.cuda.synchronize()

    start = time.time()
    for _ in range(1000):
        output2 = torch_chunk_parallel_onc(v1, v2, g1, g2, q, chunk_size=chunk_size,use_triton=True, use_cuda=False)
    torch.cuda.synchronize()
    end = time.time()
    print("gated retnet time w/o cuda", end - start)





        






    
    
    
    

        
        
        
        
        
    


    
    
    
    
    

    


    
    
    
    


    

    

    
