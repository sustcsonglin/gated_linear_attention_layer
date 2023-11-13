from triton_on2.full import flash_gret_full
from triton_onc.chunk_scan_triton_full import inter_chunk_onc
from triton_on2.naive import naive
import torch 
import torch.nn.functional as F
from einops import rearrange
import time 

if __name__ == "__main__":
    B = 2
    H = 1
    L = 2048
    D_QK = 1024
    D_V = 2048


    dtype = torch.bfloat16
    q = torch.rand(B, H, L, D_QK, device='cuda').to(dtype)
    k = torch.rand(B, H, L, D_QK, device='cuda').to(dtype)
    v = torch.rand(B, H, L, D_V, device='cuda').to(dtype)
    gk = torch.randn(B, H, L, D_QK, device='cuda').to(dtype)
    gv = torch.randn(B, H, L, D_V, device='cuda').to(dtype)

    gk = (F.logsigmoid(gk) / 16) 
    gv = (F.logsigmoid(gv) / 16) 

    for _ in range(100):
        # o = naive(q, k, v, gk, gv, BLOCK_N = 64)
        o2 = inter_chunk_onc(q, k, v, gk, gv, chunk_size = 64)
        o2 = inter_chunk_onc(q, k, v, gk, gv, chunk_size = 16)
        o2 = inter_chunk_onc(q, k, v, gk, gv, chunk_size = 32)
        o2 = inter_chunk_onc(q, k, v, gk, gv, chunk_size = 128)
        o2 = inter_chunk_onc(q, k, v, gk, gv, chunk_size = 256)
        # o2 = inter_chunk_onc(q, k, v, gk, gv, chunk_size = 512)
        # o3 = flash_gret_full(q, k, v, gk, gv, BLOCK_N = 64)
    
    print("warm up done.")

    for chunk_size in [16, 32, 64, 128, 256, 512]:
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(1000):
            o2 = inter_chunk_onc(q, k, v, gk, gv, chunk_size = chunk_size)
            # if requires_grad:
            #     output2.sum().backward(retain_graph=True)    
        torch.cuda.synchronize()
        end = time.time()
        print(f"scan gret onc, time:{end - start},  chunk size: {chunk_size}")



    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(1000):
    #     o3 = flash_gret_full(q, k, v, gk, gv, BLOCK_N = 128)
    #     # if requires_grad:
    #     #     output2.sum().backward(retain_graph=True)    
    # torch.cuda.synchronize()
    # end = time.time()
    # print("flash gret on2, require_grad:{}", end - start)




    