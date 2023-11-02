import torch
# from .triton_on2 import flash_rnn_on2
from einops import rearrange
import triton 
import triton.language as tl
from cuda import cuda_compute_intra
from pytorch_chunk_onc import torch_chunk_parallel_onc 
# from fused_onc import fused_chunk_parallel_onc
from fused_chunk_onc import fused_chunk_parallel_onc
from pytorch_chunk_onc_v2 import torch_chunk_parallel_onc_v2 
from pytorch_chunk_onc_nogv import torch_chunk_parallel_onc_nogv

from triton_flashattn import attention 


if __name__ == "__main__":
    B = 2
    H = 4
    L = 2048
    D_K = 256
    D_V = 512
    chunk_size = 16
    device = "cuda"
    requires_grad = True
    dtype = torch.bfloat16

    v1 = (torch.randn(B, H,  L, D_K)).cuda().to(dtype).requires_grad_(requires_grad)
    v2 = (torch.randn(B, H, L, D_V)).cuda().to(dtype).requires_grad_(requires_grad) 
    g1 = torch.randn(B, H,  L, D_K).cuda().uniform_(0.9, 0.99).log().to(dtype).requires_grad_(requires_grad)
    g2 = torch.randn(B, H, L, D_V).cuda().uniform_(0.9, 0.99).log().to(dtype).requires_grad_(requires_grad)
    
    q = (torch.randn(B, H, L, D_K) * 5).cuda().to(dtype).requires_grad_(requires_grad)  

    target = [v1, v2, g1, g2, q]

    import time 

    for _ in range(100):
        output2 = torch_chunk_parallel_onc(v1, v2, g1, g2, q, chunk_size=chunk_size,use_triton=True, use_cuda=True)
        if requires_grad:
            output2.sum().backward(retain_graph=True)
        # output2 = torch_chunk_parallel_onc(v1, v2, g1, g2, q, chunk_size=128,use_triton=True, use_cuda=True)
        # if requires_grad:
            # output2.sum().backward(retain_graph=True)
        
        # output2 = fused_chunk_parallel_onc(v1, v2, g1, g2, q, chunk_size=16,use_triton=True, use_cuda=True)

        output2 =  torch_chunk_parallel_onc_nogv(v1, v2, g1, q, chunk_size=64,use_triton=True, use_cuda=True)
        if requires_grad:
            output2.sum().backward(retain_graph=True)


    # print('warm up done')
    torch.cuda.synchronize()

    start = time.time()

    for _ in range(1000):
        output2 = torch_chunk_parallel_onc(v1, v2, g1, g2, q, chunk_size=chunk_size,use_triton=True, use_cuda=True)
        if requires_grad:
            output2.sum().backward(retain_graph=True)
    
    torch.cuda.synchronize()
    end = time.time()
    print("gated retnet time, require_grad:{}", end - start, requires_grad)

    # torch.cuda.synchronize()

    # start = time.time()
    # for _ in range(1000):
    #     output2 = torch_chunk_parallel_onc(v1, v2, g1, g2, q, chunk_size=256,use_triton=True, use_cuda=False)
    #     if requires_grad:
    #         output2.sum().backward(retain_graph=True)    
    # torch.cuda.synchronize()
    # end = time.time()
    # print("gated retnet time chunk 64, require_grad:{}", end - start, requires_grad)

    torch.cuda.synchronize()
    start = time.time()
    for i in range(1000):

        output2 =  torch_chunk_parallel_onc_nogv(v1, v2, g1, q, chunk_size=64,use_triton=True, use_cuda=True)
        if requires_grad:
            output2.sum().backward(retain_graph=True)

    torch.cuda.synchronize()
    end = time.time()
    print("gated retnet time chunk 64, require_grad:{}", end - start, requires_grad)




    # start = time.time()
    # for _ in range(1000):
    #     output2 = torch_chunk_parallel_onc(v1, v2, g1, g2, q, chunk_size=512,use_triton=True, use_fused_loop=True)
    #     if requires_grad:
    #         output2.sum().backward(retain_graph=True)
    


    # torch.cuda.synchronize()
    # end = time.time()
    # print("gated retnet time chunk 128 fused loop, require_grad:{}", end - start, requires_grad)




    # torch.cuda.synchronize()














        






    
    
    
    

        
        
        
        
        
    


    
    
    
    
    

    


    
    
    
    


    

    

    
